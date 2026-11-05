# HMDA Mortgage Intelligence

Historical Lending Analytics, Model Intelligence & Fair-Lending Monitoring.

A React + TypeScript + Tailwind/Recharts dashboard served by one FastAPI process. It reads the existing Domino Dataset without modifying it. This is historical research and monitoring, **not underwriting**. No individual decision endpoint is provided.

## Complete training workflow and handoff

The repository now includes a standalone ingestion → validation → training →
holdout/fairness evaluation → verified managed tracking → artifact workflow in `workflows/train_hmda.py`.
The existing dashboard, saved-model Job and MLflow comparison remain separate.
For a Domino Job with the existing Dataset attached read-only:

```bash
python workflows/train_hmda.py --tracking required
```

Use Python 3.12 with `workflows/requirements.txt`. New outputs go only to a unique
`/mnt/artifacts/hmda-training/<timestamp>-<UUID>/` directory. For local verification:
`python workflows/train_hmda.py --output-root outputs/training-runs`.
Read [the FDE handoff](docs/HANDOFF.md) for exact setup, source modes, reproducibility,
success checks and promotion boundaries, and [the source contract](docs/HMDA_SOURCE.md)
for official CFPB sources and the current HTTP 403 archive-access limitation.
No synthetic fallback is used. The original notebook remains preserved.

## Analyst copilot and current verification

The new `#analyst` source page supports aggregate questions through explicit,
read-only analytics tools. Answers include API evidence, target definitions and
limitations; the tool interface cannot score applicants or execute arbitrary code.
Domino AI Gateway runs a bounded multi-step tool loop when configured, while deterministic rendering
keeps metric values grounded in validated artifacts. With no configured endpoint,
the UI explicitly labels the agent unavailable and offers basic single-tool lookup.
The agent observes tool results, chooses follow-up tools, and finishes with validated
evidence citations; its activity trace is visible in the conversation. See
[analyst configuration](docs/HANDOFF.md#analyst-configuration-and-boundaries).

On **2026-11-03**, the complete workflow ran locally on the mounted real HMDA
sample and as Domino Job `6aea507a06b33052fec8cdfd`. Managed MLflow training run
`41be1809261343c48d2518588de89e97` finished and its values/artifacts were read back.
This is actual training, separate from the earlier retrospective comparison.
See [verification evidence and requirement status](docs/VERIFICATION.md).
Live CFPB archive access returned HTTP 403; no new source download is claimed.
The original Dataset, models, reports and deployed App were preserved. The local
frontend bundle now includes the analyst page; its predecessor is backed up under
`outputs/frontend-before-agent/`. Launch the workspace app and choose **Analyst
copilot**, or open `#analyst`. No deployment was performed by this change.

## Critical finding: actual target differs from the initial brief

Inspection of the notebook and all five Parquet files found action codes **1, 2 and 3**, not only 1 and 3. The existing `approved` label is 1 for **originated (1) OR approved but not accepted (2)**, and 0 for denied (3). The application preserves the trained pipeline and saved metrics, labels the historical target explicitly, and calculates **origination separately from action 1**. Restricting the model to actions 1 and 3 would change the evaluation; that has not been done.

2025: 25,000 records; 22,123 originated, 698 approved but not accepted, 2,179 denied. Saved-positive prevalence is 91.284%, while origination is 88.492%. An always-positive model therefore exceeds the preferred model's raw accuracy. Balanced accuracy weights both classes equally and ROC-AUC measures ranking across thresholds. Scores have not been evaluated for calibration.

## Inspection and artifact sources

Original repository: `README.md`, `HMDA Mortgage Calc.ipynb`, notebook checkpoint, and Git metadata. The notebook is preserved unchanged.

Dataset: `/mnt/data/HMDA_Project/hmda_2021.parquet` through `hmda_2025.parquet`, each 25,000 rows and 28 columns. Original models and reports:

- `models/mortgage_gradient_boosting.joblib` — present; preferred one-hot HistGradientBoosting pipeline, version 2.0.
- `models/gradient_boosting_metrics.json` — present.
- `models/mortgage_approval_model.joblib`, `models/model_metrics.json` — present; baseline.
- `reports/gradient_boosting_permutation_importance.csv` — present; validation ROC-AUC decrease, 5 repeats.
- `reports/gradient_boosting_fairness_audit_2025.csv` — **missing in the original Dataset**.
- Original baseline reports (`fairness_audit_2025.csv`, `permutation_importance.csv`, coefficients, approval rates) exist but are never substituted for preferred-model reports.

The missing preferred-model fairness report and confusion matrix were generated without training, using the saved pipeline and notebook preprocessing. They live in ignored `outputs/reports/`. All four reevaluated test metrics matched saved values to tolerance 1e-10. Evaluation JSON records model and test-data SHA-256 hashes. Dataset reports take precedence; generated reports are used only when the named Dataset report is absent. The lineage page exposes the source paths.

**Generated files are not in Git.** Recreate them in every new runtime or attach a durable report directory and set `HMDA_REPORT_DIR`. The server never loads joblib or runs predictions. Offline report generation must use trusted model files.

## Setup and local development

Use Python 3.12 and the pinned backend dependencies. Original workspace had Python 3.12.15, FastAPI, pandas, PyArrow, sklearn and joblib; Node/npm and pytest were missing. A project-local `.venv` was created with system packages inherited. Node 24.19.0/npm 11.17.0 were installed through `nodejs-wheel` for this workspace. Neither the venv nor Node is versioned.

```bash
cd /mnt/code
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r tests/requirements.txt
# If Node is absent (or provision Node 22+ in the Compute Environment):
pip install nodejs-wheel==24.19.0
npm ci --prefix frontend
python -m scripts.build_reports
python -m scripts.validate_artifacts
npm run build --prefix frontend
PORT=8000 ./app.sh
```

Open the workspace's forwarded port 8000 for the built app. VS Code may already occupy 8888 in a Workspace. App containers use 8888 by default; match the port configured for the App. There was no port override in the inspected workspace environment.

For hot reload, run these in separate terminals:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
npm run dev --prefix frontend
```

Vite listens on 5173 and proxies `api/` to port 8000. Production uses one process and one port. Assets use `base: './'`; API URLs resolve `api/` against `document.baseURI`. Hash navigation preserves Domino's proxy prefix. Use a trailing slash on the App root URL.

## Tests and API

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
npm run build --prefix frontend
python -m scripts.validate_artifacts
curl -f http://127.0.0.1:8000/api/health
curl -f http://127.0.0.1:8000/api/ready
curl -f http://127.0.0.1:8000/api/overview
```

Tests use temporary synthetic fixtures solely for validation, never as dashboard data. They never write production Dataset artifacts. Sandbox networking/event-loop restrictions can prevent TestClient from running; run tests in the ordinary workspace terminal in that case.

Endpoints: `health` (liveness), `ready` (data/report readiness), `overview`, `trends`, `distributions`, `metrics`, `evaluation`, `importance`, `fairness`, `lineage`, `filters`, all under `api/`. Interactive API documentation: `api/docs`. Analytics filters: `year`, `loan_type`, `race`; fairness: `min_samples`. Model evaluation always uses the fixed 2025 cohort. Unknown filter combinations return empty data, not invented values. Missing artifacts return 503 with an actionable error. Readiness returns degraded until all analytical artifacts are available.

Data loads once per process; artifact reads are cached where appropriate. Restart after replacing input files. Histograms retain negative and extreme values in open-ended bins and explicitly count excluded nonnumeric values. Income is in thousands of dollars; loan amounts are dollars. Missing categorical values remain a category. Full categorical distributions are scrollable; categories are not silently truncated.

## Domino Compute Environment and App deployment

**Historical setup instructions:** the owner now reports a published, healthy production App. The new training workflow does not change or redeploy it. Earlier audit results describe the then-current preview; verify current runtime readiness separately.

1. In `HMDA_Project`, launch a VS Code Workspace from the Git branch containing these files. Changes must be committed and pushed **only after owner approval** before a Git-based App can reliably publish them.
2. Select/build a Compute Environment with Python 3.12, pinned `backend/requirements.txt`, and Node 22+ (24.19 tested). Install build dependencies in the environment or allow `npm ci` at startup. The current workspace's `.venv` does not transfer to an App. A possible environment customization is `RUN pip install nodejs-wheel==24.19.0` plus the backend packages; ensure the environment's Python matches `HMDA_PYTHON`.
3. Attach the existing `HMDA_Project` Dataset to the App, read-only. Confirm its actual mount; set `HMDA_DATA_DIR` if different from `/mnt/data/HMDA_Project`.
4. Generate reports with the Job command below into a **new writable durable output location**, attach that location to the App, and set `HMDA_REPORT_DIR` accordingly. Alternatively run the documented generation command in the App preparation environment. Do not assume workspace `outputs/` appears in a new container.
5. In the project's Apps UI, create/configure the App with launch command `./app.sh`, appropriate Compute Environment, hardware, Dataset attachments, and port **8888** (or set `PORT` to the configured port). Keep access restricted for preview.
6. Start **App Preview**. Check logs, open all five pages, and visit the preview URL plus `api/ready`. Require HTTP 200 and all checks ready. This validates mount access **from the App process**, not just the Workspace.
7. Publish a named App version tied to the approved Git commit and environment. Record the Dataset snapshot/report provenance. Deploy that published version through the Apps UI after approval; recheck `api/ready` and browser requests under the deployed proxy prefix.

`app.sh` starts from its own repository directory, validates data, checks Python imports, builds from the lockfile when `frontend/dist` is absent, and runs Uvicorn on `0.0.0.0`. Set `HMDA_BUILD_FRONTEND=1` to force a rebuild; otherwise an existing tested bundle is reused. Missing optional reports leave individual pages degraded; missing/corrupt data prevents startup. No automatic package installation or report generation occurs in Python request handlers.

Configuration: `HMDA_DATA_DIR`, `HMDA_REPORT_DIR`, `HMDA_PYTHON`, `HMDA_BUILD_FRONTEND`, `PORT`, optional comma-separated `HMDA_CORS_ORIGINS` for explicit local origins. CORS is disabled by default because production is same-origin. No credentials are read or emitted by the API. Artifact paths and Git SHA are displayed intentionally; other environment variables are not exposed.

## Domino Jobs and reproducibility

Prepared commands (not submitted):

```bash
# Read-only validation Job
python -m scripts.validate_artifacts

# Offline evaluation + missing report generation, existing model only
# Use a new directory for each run; never overwrites existing reports.
python -m scripts.build_reports --output /mnt/artifacts/hmda-reports/run-001
```

For a Job, select the approved code commit, matching Compute Environment, and attached Dataset. Set `HMDA_REPORT_DIR` to the generated run directory when validating or consuming outputs. Ensure Job artifacts are persisted and accessible to the App; `/mnt/artifacts` is not implicitly shared between runs. Save the Job ID with the commit, environment version, Dataset snapshot and generated hashes. Model evaluation and report generation share one command to enforce metric verification before writing reports. No data downloads, model fitting, threshold tuning or artifact overwrites occur. The evaluation command never retrains. Use the separately documented training workflow to create a new model in an isolated run directory.

The original environment had MLflow dependency conflicts. These were subsequently resolved for the standalone retrospective comparison: two runs were logged and read back from Domino managed tracking. See the MLflow section below. The new training workflow writes isolated run artifacts and does not silently replace those existing comparison runs.

## Interpretation and limits

The notebook's sampling filters include purchase, first lien, principal residence, site-built, 1–4 units and actions 1–3. These samples are not necessarily nationally representative. Equal annual sample sizes are not application-volume estimates. The notebook explores multiple model variants/test thresholds, so the 2025 results should not be marketed as an independently locked final validation. Use a new untouched cohort for that claim.

Group comparisons are unadjusted descriptive statistics, not discrimination findings or causal estimates. Small groups and small denial denominators are flagged; null recall means no actual denials. Missing/unavailable demographic categories are retained and excluded from the named-group rate range. No uncertainty intervals or covariate controls are implemented. The range is a percentage-point spread, not a significance test.

## Files

- `backend/`: read-only services, schemas, API, pinned dependencies.
- `frontend/`: five-page responsive React dashboard and lockfile.
- `scripts/`: artifact validation and deterministic offline evaluation/report generation.
- `tests/`: temporary-fixture API and service tests.
- `app.sh`: single-process Domino entry point.
- `outputs/`: ignored generated reports and local verification artifacts.
- `HMDA Mortgage Calc.ipynb`: original, unchanged.

## Browser verification

Optional tooling (not needed to run the App):

```bash
pip install playwright==1.64.1
PLAYWRIGHT_BROWSERS_PATH=/tmp/hmda-browsers python -m playwright install chromium
# With PORT=8000 ./app.sh running:
PLAYWRIGHT_BROWSERS_PATH=/tmp/hmda-browsers python -m scripts.browser_smoke
```

The browser script checks the five pages, year and fairness filters, mobile layout/navigation, JavaScript errors, and prefix-preserving assets/API requests via a simulated prefix-stripping proxy. Screenshots go to ignored `outputs/screenshots/`. This simulation does not establish that a deployed Domino App can access its Dataset.

The production bundle currently emits Vite's size advisory (about 571 kB JavaScript / 165 kB gzip, including charts). It builds successfully; route-level splitting is a possible future optimization. The initial shared Python environment had dependency conflicts. MLflow was subsequently verified for the retrospective comparison. Use the documented pinned requirements and a versioned Compute Environment for reproducible workloads rather than assuming all workspace packages transfer to a fresh runtime.

## Private App Preview without workspace port forwarding

The workspace's advertised port-forward URL returned an ingress 404 even while the app and local code-server proxy responded successfully. Use Domino's built-in App Preview as the alternative.

`./preview.sh` is the dedicated entry point. It uses the small versioned `frontend/dist` production bundle, so Node.js is not required in the preview container. Unless `HMDA_REPORT_DIR` is explicitly configured, it creates a unique temporary report directory and evaluates the existing saved pipeline into it before starting the app. It does not train models, overwrite Dataset files, or publish an App. All required data/report checks must pass before it starts. The Compute Environment still requires `backend/requirements.txt` and the attached HMDA Dataset.

App Preview resolves the project's Git commit, rather than this workspace's uncommitted files. Commit/push requires owner approval before creating this preview. Use a review branch and a grant-based draft; do not publish publicly. Rebuild `frontend/dist` whenever frontend source changes and include the resulting assets in the approved commit. The main `app.sh` workflow remains available for environments with Node or an existing frontend build.

## Reproducible saved-model evaluation Job

In **HMDA_Project → Jobs → Run**, use this command from the repository root:

```bash
python jobs/evaluate_model.py
```

Select the Git revision containing this script and the existing Python 3.12 Compute
Environment with `backend/requirements.txt` installed (including scikit-learn 1.7.2,
joblib, pandas, PyArrow, NumPy and threadpoolctl). No Node, frontend build, package
installation at startup, model fitting, or App restart is needed. Attach the
existing Dataset at `/mnt/data/HMDA_Project`; the Job requires read access to the
saved model, `models/gradient_boosting_metrics.json`, and `hmda_2025.parquet`, plus
write access to `reports/job_runs/`. A read-only Dataset snapshot cannot accept
this report; use a writable mount. The Job also writes a copy to `/mnt/artifacts/`
(created automatically) so Domino captures it as a Job artifact. For local runs,
set `HMDA_JOB_ARTIFACT_DIR` to use a different artifact directory.

The script reuses the backend's feature order and preprocessing. It verifies the
2025 cohort and saved action mapping (1 + 2 positive, 3 denied), applies the saved
0.89 threshold, and compares all four metrics against the saved model metrics
with absolute tolerance `1e-10` and zero relative tolerance. Denial recall uses
class 0; confusion-matrix rows/columns are ordered `[0, 1]`.

If `gradient_boosting_evaluation_2025.json` exists in Dataset `reports/`, or in the
backend's `HMDA_REPORT_DIR` fallback (default workspace `outputs/reports/`), it also
compares its metrics, confusion matrix, threshold, and model/data hashes. This
supplemental report is currently absent from the Dataset; a fresh Job does not
inherit workspace outputs. Its absence is recorded explicitly, and the mandatory
saved model metrics comparison still runs. To require an existing supplemental
report, use `python jobs/evaluate_model.py --evaluation-report /path/to/existing/gradient_boosting_evaluation_2025.json`.
A missing explicitly requested report fails the Job; no reference is regenerated.

Each execution writes a new, exclusively created Dataset report and an identical
Job artifact with this filename:
`/mnt/data/HMDA_Project/reports/job_runs/evaluation_2025_<UTC timestamp>_<UUID>.json`.
It contains execution status/times, exit code, Git SHA and dirty-tree indicator,
Domino run ID, package versions, input paths/hashes, cohort details, metrics,
confusion matrix, and comparison results. Validation failures also produce a
failure report when the destination is writable. Failure to write either report
prints an error and exits nonzero. Existing dashboard reports and model/data
files are never overwritten. Dataset reports live outside the Git repository.
The artifact copy appears in the completed run's **Artifacts** view in Domino
Jobs.

Verify **Succeeded** in Domino Jobs, exit code **0**, and stdout
`HMDA 2025 evaluation: SUCCEEDED`, followed by printed Dataset report and Job
artifact paths. Open the artifact from the Job run's Artifacts view and require
`status: succeeded` and all comparison `passed` values true. Expected metrics:
ROC-AUC 0.7784548518, accuracy 0.88208, balanced accuracy 0.7071016146,
denial recall 0.4951812758; confusion matrix `[[1079,1100],[1848,20973]]`.
Use the recorded Git SHA and input hashes to identify the evaluated version.
A dirty working tree means the commit alone does not capture local source edits.

Run the focused tests with `python -m pytest -q tests/test_evaluate_model.py`.

## Domino MLflow: retrospective model comparison

Run from the repository root in the existing Domino Workspace or a Domino Job:

```bash
python experiments/log_model_comparison.py
```

The script creates the explicitly named experiment **HMDA Mortgage Approval -
Model Comparison** with **Logistic Regression - Baseline** and **Gradient Boosting
- Preferred**. It records retrospective evaluations of existing saved pipelines,
not training runs. No autologging, model registration, model binary upload, App
changes, or Dataset writes occur. It uses the same Python/backend environment as
the evaluation Job plus MLflow (3.3.2 verified here); no additional installation
is needed in the current environment.

Tracking must use Domino's injected `MLFLOW_TRACKING_URI` pointing to its local
managed tracking proxy, `http://127.0.0.1:8768` (or `localhost:8768`), with
`DOMINO_PROJECT_ID` present. The script validates service access before creating
runs. It rejects unset, file-based, or other unverified tracking URIs and never
falls back to `mlruns`. If another Domino environment uses a different managed
endpoint, verify that endpoint with the administrator before adapting the explicit
allowlist. Import/service failures exit nonzero; they are not reported as success.

Both models are reevaluated against the same existing 25,000-row
`/mnt/data/HMDA_Project/hmda_2025.parquet`, using shared backend preprocessing and
the saved target (actions 1 + 2 positive, action 3 denied). The preferred model's
validation reuses `jobs/evaluate_model.evaluate()` without writing its Job
outputs. Baseline threshold is 0.5; preferred threshold is 0.89. All four metrics
must match their respective saved JSON results with absolute tolerance `1e-10`
and zero relative tolerance **before either run is logged**. Threshold-dependent
metrics compare the saved operating points, not a common threshold or new tuning.

Runs contain feature order, model type and saved estimator parameters, evaluation
year, record count, Git SHA/dirty-tree flag, model and Dataset SHA-256, software
versions, and Domino context. Small artifacts include saved metrics, verified
evaluation evidence, and confusion matrices (rows/columns `[denied, positive]`).
The existing preferred evaluation JSON is attached when available through the
backend's Dataset-first/report-directory fallback. No raw records are uploaded.
Git metadata describes the logging execution; a dirty tree means its commit does
not fully describe the local source.

**Repeat runs:** a stable evaluation ID hashes model/data contents, threshold,
feature definitions, year, saved metrics, and evaluation protocol version.
Unchanged inputs reuse the same finished runs after validating their server-side
metrics, parameters and artifacts. Failed/incomplete runs are retried with the
same run ID. Changed inputs create new runs. Original logging Git/software
metadata is retained when a finished run is reused. Run only one comparison Job
at a time: MLflow has no atomic uniqueness constraint for tags, so concurrent
first executions are not supported. Multiple matching identities cause a clear
failure for review rather than silently selecting a run. An existing experiment
with the same name but a different project tag is rejected.

In **HMDA_Project → Experiments**, open **HMDA Mortgage Approval - Model
Comparison**, select both labeled runs, and compare ROC-AUC, accuracy, balanced
accuracy and denial recall. Check their retrospective tags and artifacts. A
successful command prints `VERIFIED`, the experiment ID and both run IDs only
after both runs are `FINISHED` and their metrics, parameters and JSON artifacts
have been read back from the tracking server. A partial logging failure exits
nonzero; rerunning repairs/reuses the existing identity without starting over.

Tests: `python -m pytest -q tests/test_model_comparison.py` (uses an in-memory
tracking double; real tracking verification is performed by the command above).

## Engineering verification

See [production-readiness evidence](docs/PRODUCTION_READINESS.md) for the modular
architecture, version-control/test evidence and remaining operational acceptance
checks. Run `bash scripts/check.sh` in the prepared verification environment;
it checks tests, TypeScript and an isolated frontend build without retraining or
deploying.

## Direct OpenAI analyst connection

The analyst can use `gpt-5.4-mini` through OpenAI without Domino AI Gateway.
Set server-side `HMDA_LLM_PROVIDER=openai`, `HMDA_OPENAI_MODEL=gpt-5.4-mini`,
and provide `OPENAI_API_KEY` as a runtime secret. The App/agent run in Domino;
questions and aggregate evidence are sent to OpenAI for inference. No additional
agent framework is required. See [setup and live verification](docs/OPENAI_ANALYST.md).
Run `python scripts/check_analyst_connection.py` after configuring the key;
configuration alone does not establish working model access. No live call was
verified during implementation because no key was available.
