# ActiveVision Human-Guidance DSS Experiment

Prolific data-collection app for the Strategic Decision Support paper.
Support modality: **human free-text guidance** on the
[ActiveVision](https://huggingface.co/datasets/activevisionai/ActiveVision) benchmark
(85 items, 17 tasks). Model: `gpt-5.6-luna`.

Unlike the shape-counting experiment, **no algorithm runs online** — this app only
collects `(x, y0, human_guidance, y1, g)` per trial. The SOS algorithm is run
offline on the exported dataset.

## Flow per trial
1. Show image + question + the model's pre-computed answer (y0).
2. Participant answers the question themselves, judges the model, writes guidance.
3. App calls `gpt-5.6-luna` live with image + question + y0 + guidance → y1.
4. Full record logged to **Redis** (primary, `/admin/export`) and **Qualtrics** (backup).

## Deploy (new Render service, same pattern as the shape-counting repo)
1. New GitHub repo with these files; connect to a **new** Render web service.
2. Start command: `gunicorn app:app --threads 8 --timeout 180`
   ⚠️ **Threads are required.** The y1 call blocks for up to ~90 s; the old
   `WEB_CONCURRENCY=1` sync-worker setup would stall every concurrent participant.
3. Attach a Redis (Key-Value) instance; can reuse the existing one — this app
   namespaces all keys under `av:` so it won't collide, but a fresh instance is cleaner.
4. Env vars: `REDIS_URL`, `ADMIN_PASSWORD`, `OPENAI_API_KEY`,
   `QUALTRICS_DATACENTER`, `QUALTRICS_API_TOKEN`, `QUALTRICS_SURVEY_ID`,
   `PROLIFIC_COMPLETION_CODE`. Optional overrides: `TRIALS_PER_PARTICIPANT` (20),
   `MODEL_NAME`, `Y1_MAX_COMPLETION_TOKENS`, `Y1_TIMEOUT_S`, `MIN_HINT_CHARS`.

## Launch checklist
**Before publishing on Prolific**
- [ ] Run the Colab notebook Cells 1–11 → generate real `y0_data.json` for all 85
      items → replace the placeholder in the repo root → commit & push.
- [ ] Check `/health` (should report `n_items: 85`).
- [ ] `POST /admin/reset_all` with `{"password": "..."}` to clear any test state.
- [ ] Do one full 20-trial pass yourself; verify rows land in Qualtrics AND in
      `/admin/export?password=...`.
- [ ] Free tier spins down — hit the URL once right before launch.

**Qualtrics** — create a **new project** (per the plan to keep data separate).
Add embedded-data fields matching the record keys in `app.py` (`participant_id`,
`trial_overall`, `item_id`, `task`, `ground_truth`, `y0_answer`, `y0_correct`,
`human_answer`, `human_answer_correct`, `human_says_model_correct`,
`human_hint_text`, `y1_answer`, `y1_correct`, `g`, `answer_changed`, `timestamp`, ...).
Note: Qualtrics may truncate long text fields (`y0_raw`, `y1_raw`, long hints) —
Redis `/admin/export` is the authoritative dataset.

**Prolific** — create a **new study** pointing at the new Render URL with
`?PROLIFIC_PID={{%PROLIFIC_PID%}}`. For 1000 trials at 20 trials/participant:
**50 places** (each of the 85 items seen ~12×). Budget ~30 min per participant.

**After collection** — Colab Cells 12–14: pull `/admin/export`, dedupe, recompute
scoring, save `activevision_dss_trials.csv` for the offline SOS run.

## Cost & timing estimate (sanity-check in Colab Cell 7 first)
1000 live y1 calls + 85×(y0 + optional g_hat + decision) calls. Each call sends a
~1536×1024 image and can produce thousands of reasoning tokens — check the usage
numbers Cell 7 prints and multiply before launching.
