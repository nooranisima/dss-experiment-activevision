"""
ActiveVision Human-AI Collaboration Experiment (Prolific data collection)
=========================================================================
Strategic Decision Support paper — human-guidance support modality.

Flow per trial:
  1. Show the ActiveVision image + question + the model's initial answer (y0,
     pre-computed offline and baked into y0_data.json).
  2. Participant inspects the image, records their OWN answer, states whether
     they think the model is right, and writes free-text guidance for the model.
  3. Server calls the model live (gpt-5.6-luna) with the image + question +
     its previous answer + the human's guidance -> y1.
  4. Everything is logged to Redis (primary backup) and Qualtrics (per-trial
     response). No online algorithm runs here — analysis is offline.

Env vars (Render dashboard):
  REDIS_URL, ADMIN_PASSWORD, OPENAI_API_KEY,
  QUALTRICS_DATACENTER (e.g. "iad1" or full "https://iad1.qualtrics.com"),
  QUALTRICS_API_TOKEN, QUALTRICS_SURVEY_ID,
  PROLIFIC_COMPLETION_CODE (optional), PROLIFIC_STUDY_ID + PROLIFIC_API_TOKEN (optional)

Run: gunicorn app:app --threads 8 --timeout 180
  (threads matter: the y1 call blocks for up to ~90s; a single sync worker
   would stall all concurrent participants.)
"""

import json
import os
import random
import re
import threading
import time
import uuid

import redis
import requests
from flask import Flask, redirect, render_template_string, request, session, url_for

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL = os.environ.get("MODEL_NAME", "gpt-5.6-luna")
TRIALS_PER_PARTICIPANT = int(os.environ.get("TRIALS_PER_PARTICIPANT", "20"))
Y1_MAX_COMPLETION_TOKENS = int(os.environ.get("Y1_MAX_COMPLETION_TOKENS", "4000"))
Y1_TIMEOUT_S = int(os.environ.get("Y1_TIMEOUT_S", "150"))
MIN_HINT_CHARS = int(os.environ.get("MIN_HINT_CHARS", "25"))

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PROLIFIC_COMPLETION_CODE = os.environ.get("PROLIFIC_COMPLETION_CODE", "")

QUALTRICS_DATACENTER = os.environ.get("QUALTRICS_DATACENTER", "")
QUALTRICS_API_TOKEN = os.environ.get("QUALTRICS_API_TOKEN", "")
QUALTRICS_SURVEY_ID = os.environ.get("QUALTRICS_SURVEY_ID", "")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", uuid.uuid4().hex)

# ----------------------------------------------------------------------------
# Redis
# ----------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL, decode_responses=True)

K_EXPOSURE = "av:exposure:{item_id}"          # int, times a trial was SUBMITTED
K_PENDING = "av:pending:{item_id}"            # int w/ TTL, in-flight assignments
PENDING_TTL_S = int(os.environ.get("PENDING_TTL_S", "3600"))
K_PARTICIPANT = "av:participant:{pid}"        # json state
K_LOG = "av:log:{pid}:{trial}"                # json full trial record
K_LOG_INDEX = "av:log_index"                  # list of log keys
K_TRIAL_COUNTER = "av:global_trial_counter"   # int


# ----------------------------------------------------------------------------
# Item bank (pre-computed y0)
# ----------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(__file__), "y0_data.json")) as f:
    ITEMS = json.load(f)
ITEM_BY_ID = {it["id"]: it for it in ITEMS}
print(f"Loaded {len(ITEMS)} items with pre-computed y0 (model={MODEL})")


# ----------------------------------------------------------------------------
# Answer parsing / scoring (matches the ActiveVision eval protocol:
# last <answer>...</answer> block, exact match after normalizing case,
# whitespace, and separators; integers also match numerically)
# ----------------------------------------------------------------------------
def extract_answer(raw: str) -> str:
    if not raw:
        return ""
    blocks = re.findall(r"<answer>(.*?)</answer>", raw, re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0]
    return raw.strip().splitlines()[-1].strip() if raw.strip() else ""


def normalize_answer(ans: str, task: str = "") -> str:
    if ans is None:
        return ""
    s = re.sub(r"[^a-z0-9]", "", str(ans).strip().lower())
    if task == "maze_path_tracing" and len(s) == 2 and s.isalpha():
        s = "".join(sorted(s))  # E-F == F-E
    if s.isdigit():
        s = str(int(s))  # "07" == "7"
    return s


def check_answer(predicted: str, gold: str, task: str = "") -> bool:
    if predicted is None or gold is None:
        return False
    return normalize_answer(predicted, task) == normalize_answer(gold, task)


# ----------------------------------------------------------------------------
# y1: live model call with human guidance
# ----------------------------------------------------------------------------
Y1_INSTRUCTIONS = (
    "You previously attempted this visual question and gave the answer shown "
    "below. A human collaborator has now carefully inspected the same image and "
    "written guidance for you. The human may point out errors, describe what "
    "they see, explain their reasoning, or state what they believe the correct "
    "answer is. Use their guidance together with your own inspection of the "
    "image to produce your best final answer.\n\n"
    "Provide your final answer inside <answer></answer> tags."
)


def get_y1(item: dict, hint_text: str) -> dict:
    """Call the model with image + question + y0 + human guidance."""
    user_content = [
        {"type": "text", "text": item["question"]},
        {"type": "image_url", "image_url": {"url": item["image_url"]}},
        {
            "type": "text",
            "text": (
                f"{Y1_INSTRUCTIONS}\n\n"
                f"--- Your previous answer ---\n{item['y0_raw']}\n\n"
                f"--- Human collaborator's guidance ---\n{hint_text}"
            ),
        },
    ]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "max_completion_tokens": Y1_MAX_COMPLETION_TOKENS,
    }
    t0 = time.time()
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json=payload,
        timeout=Y1_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = (data["choices"][0]["message"]["content"] or "").strip()
    return {
        "y1_raw": raw,
        "y1_answer": extract_answer(raw),
        "y1_latency_s": round(time.time() - t0, 2),
        "y1_usage": data.get("usage", {}),
        "y1_error": "",
    }


# ----------------------------------------------------------------------------
# Qualtrics logging (per-trial response via API v3)
# ----------------------------------------------------------------------------
def _qualtrics_base() -> str:
    dc = QUALTRICS_DATACENTER.strip().rstrip("/")
    if not dc:
        return ""
    if not dc.startswith("http"):
        dc = f"https://{dc}.qualtrics.com"
    return dc


def log_to_qualtrics(values: dict):
    base = _qualtrics_base()
    if not base:
        print("[qualtrics] SKIPPED: QUALTRICS_DATACENTER is empty"); return
    if not QUALTRICS_API_TOKEN:
        print("[qualtrics] SKIPPED: QUALTRICS_API_TOKEN is empty"); return
    if not QUALTRICS_SURVEY_ID:
        print("[qualtrics] SKIPPED: QUALTRICS_SURVEY_ID is empty"); return
    try:
        # Qualtrics embedded-data values must be scalars
        clean = {k: str(v) for k, v in values.items()}  # Qualtrics requires all-string values
        resp = requests.post(
            f"{base}/API/v3/surveys/{QUALTRICS_SURVEY_ID}/responses",
            headers={"X-API-TOKEN": QUALTRICS_API_TOKEN, "Content-Type": "application/json"},
            json={"values": clean},
            timeout=20,
        )
        if resp.status_code == 200:
            print(f"[qualtrics] OK trial={values.get('trial_overall')} "
                  f"id={resp.json().get('result', {}).get('responseId', '?')}")
        else:
            print(f"[qualtrics] FAILED {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"[qualtrics] EXCEPTION: {e}")

def log_trial(record: dict):
    """Redis is the primary store; Qualtrics is logged async as secondary."""
    key = K_LOG.format(pid=record["participant_id"], trial=record["trial_overall"])
    r.set(key, json.dumps(record))
    r.rpush(K_LOG_INDEX, key)
    log_to_qualtrics(record)


# ----------------------------------------------------------------------------
# Assignment: least-exposed-first so all 85 items accumulate coverage evenly
# ----------------------------------------------------------------------------
def assign_items(n: int) -> list:
    """Pick the n items with the fewest (submitted + in-flight) trials.

    Exposure is committed only on trial SUBMIT (see /submit), so a participant
    who starts and abandons does not permanently consume items. Their in-flight
    claim is tracked in a pending counter that expires after PENDING_TTL_S,
    after which the items become fully assignable again."""
    load = []
    for it in ITEMS:
        committed = int(r.get(K_EXPOSURE.format(item_id=it["id"])) or 0)
        pending = int(r.get(K_PENDING.format(item_id=it["id"])) or 0)
        load.append((committed + pending, random.random(), it["id"]))
    load.sort()
    chosen = [item_id for _, _, item_id in load[:n]]
    random.shuffle(chosen)
    for item_id in chosen:
        pk = K_PENDING.format(item_id=item_id)
        r.incr(pk)
        r.expire(pk, PENDING_TTL_S)
    return chosen


def get_state(pid: str):
    raw = r.get(K_PARTICIPANT.format(pid=pid))
    return json.loads(raw) if raw else None


def save_state(pid: str, state: dict):
    r.set(K_PARTICIPANT.format(pid=pid), json.dumps(state))


# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------
BASE_CSS = """
:root { --ink:#1c2733; --muted:#5b6b7b; --line:#d8dfe6; --bg:#f2f5f8;
        --card:#ffffff; --accent:#0b6e6e; --warn:#8a4b08; }
* { box-sizing:border-box; }
body { margin:0; font-family:"Segoe UI",system-ui,-apple-system,sans-serif;
       background:var(--bg); color:var(--ink); line-height:1.5; }
.wrap { max-width:980px; margin:0 auto; padding:24px 16px 64px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:24px; margin-bottom:18px; }
h1 { font-size:1.35rem; margin:0 0 8px; } h2 { font-size:1.05rem; margin:0 0 10px; }
.tag { display:inline-block; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase;
       color:var(--accent); border:1px solid var(--accent); border-radius:999px;
       padding:2px 10px; margin-bottom:10px; }
.muted { color:var(--muted); font-size:.92rem; }
.qbox { background:#f7f9fb; border-left:4px solid var(--accent); padding:12px 14px;
        border-radius:0 8px 8px 0; white-space:pre-wrap; }
.aibox { background:#fdf6ec; border-left:4px solid var(--warn); padding:12px 14px;
         border-radius:0 8px 8px 0; }
.aibox .big { font-size:1.25rem; font-weight:700; }
img.stim { width:100%; border:1px solid var(--line); border-radius:8px;
           background:#fff; cursor:zoom-in; }
label { display:block; font-weight:600; margin:16px 0 6px; }
input[type=text], textarea { width:100%; padding:10px 12px; border:1px solid var(--line);
        border-radius:8px; font-size:1rem; font-family:inherit; }
textarea { min-height:130px; resize:vertical; }
.radio-row label { display:inline-block; font-weight:400; margin:0 18px 0 4px; }
button { background:var(--accent); color:#fff; border:0; border-radius:8px;
         padding:12px 26px; font-size:1rem; font-weight:600; cursor:pointer; }
button:disabled { opacity:.5; cursor:wait; }
.progress { font-variant-numeric:tabular-nums; color:var(--muted); margin-bottom:12px; }
.spinner { display:none; margin-top:14px; color:var(--muted); }
.err { color:#a01818; font-weight:600; }
ol li { margin-bottom:6px; }
"""

LANDING_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guide the AI — visual reasoning study</title><style>{{ css }}</style></head><body>
<div class="wrap">
  <div class="card">
    <span class="tag">Research study</span>
    <h1>Help an AI model answer hard visual questions</h1>
    <p>In this study you will see <strong>{{ n_trials }} images</strong>, each with a
    visual question (counting objects, tracing paths, spotting differences).
    An AI model has already attempted each question — and it is often wrong.</p>
    <p><strong>Your job:</strong> inspect the image yourself, then write guidance that
    helps the model reach the correct answer. After each round, the model re-answers
    using your guidance.</p>
    <h2>On each trial you will:</h2>
    <ol>
      <li>Study the image and answer the question yourself.</li>
      <li>Say whether you think the AI's answer is right or wrong.</li>
      <li>Write guidance for the AI: what you believe the correct answer is and the
          reasoning or visual evidence behind it (e.g. "trace the rope from the left —
          it crosses under twice, so there are 12 loops, not 9").</li>
    </ol>
    <p class="muted">Your goal is for the model's <em>revised</em> answer to be correct.
    The clearer and more specific your guidance, the better it does.
    Please do not use any external tools — just your own eyes.
    The study takes roughly 25–35 minutes. Waiting a few seconds after each
    submission is normal: the model is re-answering live.</p>
    <form method="post" action="{{ url_for('start') }}">
      <label for="pid">Prolific ID</label>
      <input type="text" id="pid" name="pid" value="{{ pid or '' }}" required
             pattern="[A-Za-z0-9]{5,64}">
      {% if error %}<p class="err">{{ error }}</p>{% endif %}
      <p><button type="submit">Begin study</button></p>
    </form>
  </div>
</div></body></html>
"""

TRIAL_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial {{ trial_num }} of {{ n_trials }}</title><style>{{ css }}</style></head><body>
<div class="wrap">
  <p class="progress">Trial {{ trial_num }} / {{ n_trials }}</p>

  <div class="card">
    <h2>The question</h2>
    <div class="qbox">{{ question }}</div>
  </div>

  <div class="card">
    <h2>The image</h2>
    <a href="{{ image_url }}" target="_blank" rel="noopener">
      <img class="stim" src="{{ image_url }}" alt="Study the image to answer the question">
    </a>
    <p class="muted">Click the image to open it full size in a new tab.</p>
  </div>

  <div class="card">
    <h2>The AI model's current answer</h2>
    <div class="aibox">
      <div class="big">{{ y0_answer }}</div>
      {% if y0_reasoning %}<details><summary class="muted">Show the model's full response</summary>
      <div style="white-space:pre-wrap; margin-top:8px;">{{ y0_reasoning }}</div></details>{% endif %}
    </div>
  </div>

  <div class="card">
    <h2>Your turn</h2>
    <form method="post" action="{{ url_for('submit') }}" id="trialform">
      <label for="human_answer">1&nbsp;·&nbsp;What do <em>you</em> think the correct answer is?</label>
      <input type="text" id="human_answer" name="human_answer" required
             placeholder="Your own answer to the question above">

      <label>2&nbsp;·&nbsp;Is the AI's answer correct?</label>
      <div class="radio-row">
        <input type="radio" id="mc_yes" name="model_correct" value="yes" required>
        <label for="mc_yes">Yes, it looks right</label>
        <input type="radio" id="mc_no" name="model_correct" value="no">
        <label for="mc_no">No, it's wrong</label>
        <input type="radio" id="mc_unsure" name="model_correct" value="unsure">
        <label for="mc_unsure">Not sure</label>
      </div>

      <label for="hint_text">3&nbsp;·&nbsp;Write your guidance for the AI</label>
      <p class="muted" style="margin:0 0 6px">State what you believe the correct answer is and
      <strong>how you arrived at it</strong> — describe what to look at in the image, where the
      model likely went wrong, and the steps of your reasoning. If you think the model is
      already right, explain why you agree. (At least {{ min_chars }} characters.)</p>
      <textarea id="hint_text" name="hint_text" required minlength="{{ min_chars }}"></textarea>

      <p><button type="submit" id="gobtn">Send guidance to the AI</button></p>
      <p class="spinner" id="spin">⏳ The model is re-answering with your guidance —
      this can take up to a minute. Please keep this tab open.</p>
    </form>
  </div>
</div>
<script>
document.getElementById('trialform').addEventListener('submit', function () {
  document.getElementById('gobtn').disabled = true;
  document.getElementById('spin').style.display = 'block';
});
</script>
</body></html>
"""

DONE_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Study complete</title><style>{{ css }}</style></head><body>
<div class="wrap"><div class="card">
  <span class="tag">Complete</span>
  <h1>Thank you — all {{ n_trials }} trials are done</h1>
  {% if code %}
  <p>Your Prolific completion code:</p>
  <div class="aibox"><div class="big">{{ code }}</div></div>
  <p><a href="https://app.prolific.com/submissions/complete?cc={{ code }}">
     Return to Prolific and submit</a></p>
  {% else %}
  <p>You may now return to Prolific to complete your submission.</p>
  {% endif %}
</div></div></body></html>
"""


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def landing():
    pid = request.args.get("PROLIFIC_PID", "")
    return render_template_string(
        LANDING_HTML, css=BASE_CSS, pid=pid, n_trials=TRIALS_PER_PARTICIPANT, error=None
    )


@app.route("/start", methods=["POST"])
def start():
    pid = (request.form.get("pid") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]{5,64}", pid):
        return render_template_string(
            LANDING_HTML, css=BASE_CSS, pid=pid, n_trials=TRIALS_PER_PARTICIPANT,
            error="Please enter a valid Prolific ID.",
        )
    state = get_state(pid)
    if state is None:
        state = {
            "items": assign_items(TRIALS_PER_PARTICIPANT),
            "trial_idx": 0,
            "started_at": time.time(),
            "completed": False,
        }
        save_state(pid, state)
    session["pid"] = pid
    return redirect(url_for("done") if state["completed"] else url_for("trial"))


@app.route("/trial")
def trial():
    pid = session.get("pid")
    if not pid:
        return redirect(url_for("landing"))
    state = get_state(pid)
    if state is None:
        return redirect(url_for("landing"))
    if state["trial_idx"] >= len(state["items"]):
        return redirect(url_for("done"))
    item = ITEM_BY_ID[state["items"][state["trial_idx"]]]
    session["trial_shown_at"] = time.time()
    return render_template_string(
        TRIAL_HTML,
        css=BASE_CSS,
        trial_num=state["trial_idx"] + 1,
        n_trials=len(state["items"]),
        question=item["question"],
        image_url=item["image_url"],
        y0_answer=item["y0_answer"] or "(no clear answer given)",
        y0_reasoning=item.get("y0_raw", ""),
        min_chars=MIN_HINT_CHARS,
    )


@app.route("/submit", methods=["POST"])
def submit():
    pid = session.get("pid")
    if not pid:
        return redirect(url_for("landing"))
    state = get_state(pid)
    if state is None or state["trial_idx"] >= len(state["items"]):
        return redirect(url_for("done"))

    item = ITEM_BY_ID[state["items"][state["trial_idx"]]]
    human_answer = (request.form.get("human_answer") or "").strip()
    model_correct = (request.form.get("model_correct") or "").strip()
    hint_text = (request.form.get("hint_text") or "").strip()
    shown_at = session.get("trial_shown_at", time.time())

    # y1: live call with the human's guidance
    try:
        y1 = get_y1(item, hint_text)
    except Exception as e:
        y1 = {"y1_raw": "", "y1_answer": "", "y1_latency_s": -1,
              "y1_usage": {}, "y1_error": str(e)[:500]}

    trial_overall = r.incr(K_TRIAL_COUNTER)
    record = {
        # identifiers
        "participant_id": pid,
        "trial_in_session": state["trial_idx"] + 1,
        "trial_overall": trial_overall,
        "timestamp": time.time(),
        "model": MODEL,
        # item / x
        "item_id": item["id"],
        "task": item["task"],
        "category": item["category"],
        "question": item["question"],
        "image_url": item["image_url"],
        "ground_truth": item["answer"],
        # y0 (pre-computed)
        "y0_answer": item["y0_answer"],
        "y0_raw": item["y0_raw"],
        "y0_correct": int(check_answer(item["y0_answer"], item["answer"], item["task"])),
        # human
        "human_answer": human_answer,
        "human_answer_correct": int(check_answer(human_answer, item["answer"], item["task"])),
        "human_says_model_correct": model_correct,
        "human_hint_text": hint_text,
        "human_time_s": round(time.time() - shown_at, 1),
        # y1 (live)
        "y1_answer": y1["y1_answer"],
        "y1_raw": y1["y1_raw"],
        "y1_correct": int(check_answer(y1["y1_answer"], item["answer"], item["task"])),
        "y1_latency_s": y1["y1_latency_s"],
        "y1_error": y1["y1_error"],
        "y1_usage_json": json.dumps(y1.get("y1_usage", {})),
    }
    record["g"] = int((not record["y0_correct"]) and record["y1_correct"])
    record["answer_changed"] = int(
        normalize_answer(record["y0_answer"], item["task"])
        != normalize_answer(record["y1_answer"], item["task"])
    )
    log_trial(record)

    # Commit this item's exposure now that the trial actually happened,
    # and release its in-flight claim.
    r.incr(K_EXPOSURE.format(item_id=item["id"]))
    pk = K_PENDING.format(item_id=item["id"])
    if int(r.get(pk) or 0) > 0:
        r.decr(pk)

    state["trial_idx"] += 1
    if state["trial_idx"] >= len(state["items"]):
        state["completed"] = True
    save_state(pid, state)
    return redirect(url_for("done") if state["completed"] else url_for("trial"))


@app.route("/done")
def done():
    return render_template_string(
        DONE_HTML, css=BASE_CSS, code=PROLIFIC_COMPLETION_CODE,
        n_trials=TRIALS_PER_PARTICIPANT,
    )


# ----------------------------------------------------------------------------
# Health & admin
# ----------------------------------------------------------------------------
def _auth_ok(req) -> bool:
    pw = req.args.get("password") or (req.get_json(silent=True) or {}).get("password")
    return bool(pw) and pw == ADMIN_PASSWORD


@app.route("/health")
def health():
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        "model": MODEL,
        "n_items": len(ITEMS),
        "trials_logged": int(r.get(K_TRIAL_COUNTER) or 0),
    }


@app.route("/admin/state")
def admin_state():
    if not _auth_ok(request):
        return {"error": "unauthorized"}, 403
    exposures = {it["id"]: int(r.get(K_EXPOSURE.format(item_id=it["id"])) or 0) for it in ITEMS}
    pendings = {it["id"]: int(r.get(K_PENDING.format(item_id=it["id"])) or 0) for it in ITEMS}
    return {
        "trials_logged": int(r.get(K_TRIAL_COUNTER) or 0),
        "n_log_entries": r.llen(K_LOG_INDEX),
        "exposure_min": min(exposures.values()) if exposures else 0,
        "exposure_max": max(exposures.values()) if exposures else 0,
        "items_with_zero_submits": [k for k, v in exposures.items() if v == 0],
        "exposures": exposures,
        "pending": {k: v for k, v in pendings.items() if v > 0},
    }


@app.route("/admin/export")
def admin_export():
    """Dump every logged trial as JSON — the primary dataset for offline analysis."""
    if not _auth_ok(request):
        return {"error": "unauthorized"}, 403
    keys = r.lrange(K_LOG_INDEX, 0, -1)
    records = []
    for k in keys:
        raw = r.get(k)
        if raw:
            records.append(json.loads(raw))
    return {"n": len(records), "records": records}


@app.route("/admin/reset_all", methods=["POST"])
def admin_reset_all():
    if not _auth_ok(request):
        return {"error": "unauthorized"}, 403
    deleted = 0
    for pattern in ["av:exposure:*", "av:pending:*", "av:participant:*", "av:log:*"]:
        for k in r.scan_iter(pattern):
            r.delete(k)
            deleted += 1
    r.delete(K_LOG_INDEX)
    r.delete(K_TRIAL_COUNTER)
    return {"reset": True, "keys_deleted": deleted}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
