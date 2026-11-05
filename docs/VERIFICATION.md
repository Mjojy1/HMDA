# HMDA analyst and managed training verification — 2026-11-03

## Scope and preservation

Initial `git status --short` was clean. Existing APIs, five dashboard pages,
notebook, Dataset, models, reports, retrospective comparison runs, and versioned
frontend bundle were preserved. Changes add an analyst page and API routes,
managed tracking in the actual training workflow, ingestion schema/seed metadata,
focused tests and documentation. Nothing was published, deployed, registered or
promoted into the live App. New generated files are isolated under `outputs/` or
unique Domino Job artifact directories. All five mounted input hashes were checked
against the new Job provenance and are unchanged.

## Requirement status

| Requirement | Status and evidence |
|---|---|
| 1. Data | Existing bounded CFPB CSV ZIP ingestion and validated mounted inputs retained. Exact input/output SHA-256, source URLs, schema dtypes, cohort counts and sampling metadata recorded. Live 2025 archive ranged GET returned HTTP 403; no new download or synthetic fallback. Mounted 2022–2025 seed 42 comes from the upstream manifest; the 2021 sampling seed is unknown, explicitly null. |
| 2. Target | Preserved 1 = originated, 2 = approved but not accepted, 3 = denied. Binary approval is 1+2 versus 3; origination is separately action 1. Tests and analyst definitions enforce the distinction. |
| 3. Training | 2021–2023 training, 2024 validation and threshold selection, 2025 holdout. Training-only imputation/encoding; six existing predictors exclude protected attributes. Parameters, seed, threshold grid, all metrics and input/model/code hashes retained. |
| 4. Managed tracking | Actual training Job succeeded. Experiment 2, run `41be1809261343c48d2518588de89e97`, `FINISHED`. Parameters and validation/holdout metrics read back; all 11 logged artifacts downloaded and hashed. Distinct training experiment; no retrospective run relabeled, no local fallback. |
| 5. Responsible AI | New training report covers race, ethnicity and sex with support, outcomes, prediction rates, recalls/errors, null denominators and supported max-minus-min disparities. Small/unavailable groups flagged. Existing live model's narrower race prediction report remains explicitly distinguished from new-model results. |
| 6. Agent + UI | New source `#analyst` page, allowlisted read-only tools, deterministic evidence rendering, definitions, refusals and loading/error/unavailable states. Original APIs preserved; added `/api/analyst`, `/api/analyst/status`, `/api/fairness/groups`. Gateway selection tested with controlled responses; live endpoint list was empty, so real Gateway inference is unverified. Clearly labeled local topic selection works. |
| 7. Domino workflow | New Job `6aea507a06b33052fec8cdfd` succeeded with retained isolated artifacts and verified MLflow. CPU training with two-thread limits; no distributed infrastructure added. Setup/replay commands in HANDOFF.md. |

All results are limited, unadjusted research on filtered samples, not proof of
discrimination or legal compliance. No individual lending decisions. The original
notebook already examined 2025, so this is not a newly untouched external holdout.

## Local run

Command:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python workflows/train_hmda.py \
  --input-dir /mnt/data/HMDA_Project \
  --output-root outputs/training-runs --tracking off
```

Artifact directory:
`outputs/training-runs/20261103T212135524782Z-22c74c9c7235485a98dae49af8826429/`

75,000 training records, 25,000 validation records, 25,000 holdout records.
Threshold **0.89**, selected by validation balanced accuracy, ties choosing the
lowest grid value. No holdout selection/refit.

| Metric | Validation 2024 | Holdout 2025 |
|---|---:|---:|
| ROC-AUC | 0.7799767358 | 0.7766438824 |
| Accuracy | 0.8715200000 | 0.8692000000 |
| Balanced accuracy | 0.7177308394 | 0.7062733114 |
| Denial recall | 0.5295404814 | 0.5089490592 |
| Approval recall | 0.9059211974 | 0.9035975636 |
| Brier score | 0.0595526869 | 0.0579338688 |
| Log loss | 0.2283199795 | 0.2234469348 |

Full rates, matrices, fairness, threshold selection and provenance are in the run
reports. Values measure approval (1+2), not borrower acceptance or origination.

## Domino execution and independent readback

- Job ID: `6aea507a06b33052fec8cdfd`, final status **Succeeded**.
- Experiment: `HMDA Training - 6ae4dc0f06b33052fec8bf56`, ID `2`.
- MLflow training run: `41be1809261343c48d2518588de89e97`, **FINISHED**.
- Job artifact directory:
  `/mnt/artifacts/hmda-training/20261103T212232389457Z-62d60831adc243d48ded45f98abf75ca/`.
- Workflow's own readback verified every logged parameter/metric and artifact
  SHA-256 before reporting success. A separate post-Job client downloaded all
  11 artifacts, checked model hash against evaluation, matched server metrics
  to evaluation JSON, and verified mounted inputs against recorded hashes.
- Local and Domino validation/holdout metrics match. Serialized model bytes differ
  across execution contexts; both models produce **identical holdout probabilities**.
  Exact binary equivalence is not claimed.

The Job executed the current reviewed workflow files as a compressed code bundle
in a new temporary Job directory, avoiding a Git push, deployment or Dataset write.
Bundle SHA-256: `88d14bed479c215d29dc57139f535eea2ed0d81f823b53601d7ff40e9a936658`.
The command ran `python workflows/train_hmda.py --tracking required` from that
isolated directory. It used the current Domino environment and existing project
Dataset mount, with Dataset/NetApp snapshot-on-completion disabled. Code provenance
therefore has no Git checkout SHA; exact per-file hashes and `code.zip` identify
the executed source. The local run records repository HEAD and dirty state.
Future routine Jobs should select the approved code revision and versioned
Compute Environment as described in HANDOFF.md.

Local verification artifacts (ignored by Git):

- `outputs/verification-20261103/managed-training-verification.json`
- `outputs/verification-20261103/remote-training-provenance.json`
- `outputs/verification-20261103/remote-artifacts/`
- `outputs/verification-20261103/job-status.json`
- `outputs/verification-20261103/source-access.json`
- `outputs/verification-20261103/screenshots/`

Use the managed experiment and retained Job artifacts for durable evidence;
workspace `outputs/` is local evidence, not a shared App mount.

## Tests and frontend

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q
PATH=/mnt/code/.venv/bin:$PATH npm run build --prefix frontend \
  -- --outDir /tmp/hmda-analyst-build --emptyOutDir
```

**81 tests passed**, including the existing suite and new agent boundaries,
Gateway invalid-response/timeout paths, training MLflow server readback corruption,
missing tracking, ingestion/target/preprocessing and fairness edge cases.
One upstream dateutil deprecation warning; no failing tests.
TypeScript and Vite production build passed; Vite retains its bundle-size advisory.

Playwright checked all six pages, aggregate analyst evidence, individual-scoring
refusal, existing filters, mobile navigation/overflow and prefix-stripping proxy
URLs. No JavaScript errors or failed HTTP responses. The smoke test used the
new temporary build and a localhost-only FastAPI server; it did not serve or
modify the live App. Chromium was downloaded to `/tmp/hmda-playwright` for this
check. `scripts/browser_smoke.py` contains the repeatable browser assertions.

## External limitations and remaining release steps

1. CFPB archive access returned HTTP 403. Mounted real samples were processed,
   but archive bytes and upstream source claims could not be freshly authenticated.
   Resolve access and use `--source cfpb --allow-download` for a fresh verified fetch.
2. No Domino Gateway endpoint was available. Provision/select an approved endpoint,
   configure server-side URL/model/credentials, and test a real aggregate question
   plus an invalid request before claiming live LLM verification.
3. The source UI is implemented and tested; the existing versioned bundle and live
   App remain unchanged. An approved release must rebuild the frontend bundle.
   Promotion of any new model/Dataset or deployment still requires explicit approval.

## Follow-up: genuine agent loop and visible local bundle

The first implementation was a single-step topic router, not a complete agentic
loop. That gap was corrected in `backend/analyst_agent.py`. The Gateway now uses
native `read_hmda` and `finish_analysis` function calls in a bounded observe/act
loop: it sees actual tool results, chooses subsequent reads, and finishes with
validated evidence IDs. The renderer combines the cited results without accepting
LLM-authored metric values or unsupported narrative facts. Tool errors can guide
subsequent actions but cannot be hidden by a successful final claim. There are
limits on steps, time, repeated calls, tool arguments and citation identities.

The conversation displays the activity trace. Six bounded prior user questions
provide context. The UI explicitly distinguishes **Agent unavailable** from the
basic single-tool lookup fallback. Live Gateway inference is still unverified
because no endpoint was configured; automated scripted Gateway responses verify
the multi-step behavior, result feedback, errors, budgets and evidence boundaries.

The local production bundle was rebuilt into `frontend/dist`, making **Analyst
copilot** (`#analyst`) available via the existing workspace app entry point. Its
previous bundle is retained under `outputs/frontend-before-agent/` and in Git.
This supersedes the earlier note that the versioned bundle was unchanged; the
live deployed App and model artifacts remain unchanged.

Follow-up validation: **90 tests passed**; TypeScript/Vite production build passed.
Chromium smoke checks passed against the rebuilt local bundle for all six pages,
analyst evidence/refusal, filters, mobile layout/navigation and proxy-prefixed URLs.
