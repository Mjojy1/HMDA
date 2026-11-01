# HMDA FDE handoff

## Existing capabilities to preserve

The React/FastAPI dashboard, read-only analytics APIs, saved-model evaluation Job,
and Domino MLflow model-comparison script already exist. The owner reports the
production App is healthy and the evaluation Job succeeded. The earlier session
also verified two retrospective MLflow runs. This work does not republish the
App, change its entry points/bundle/APIs, or replace its trusted model/report
inputs. The presentation is out of scope per the owner's follow-up; it is not
replaced or reviewed here.

## New end-to-end command

Use Python 3.12 with `workflows/requirements.txt` installed in the Compute
Environment (pin/version it in Domino). These packages are already available in
the current Workspace; no installation was performed. The requirements include
the backend dependencies plus pinned NumPy, SciPy, threadpoolctl, and requests.
Node is unnecessary for training. Use a CPU tier with at least 4 GB memory for the
existing 125,000-row sample; thread pools are limited to two workers.

In **HMDA_Project → Jobs → Run**, select the approved Git revision containing
`workflows/`, attach the existing HMDA Dataset read-only, and enter:

```bash
python workflows/train_hmda.py
```

Defaults: `/mnt/data/HMDA_Project` as input and
`/mnt/artifacts/hmda-training/<UTC timestamp>-<UUID>/` as a new output directory.
No credentials or project-name edits are needed. Ensure this Job's environment
provides writable `/mnt/artifacts` and that completed Job artifacts are retained
according to your Domino setup. It is not a shared mount inherited by the App.
The new training Job has **not** been submitted by this implementation.

For an ordinary local run, provide the input and writable output directories:

```bash
python workflows/train_hmda.py --input-dir /path/to/HMDA_Project --output-root outputs/training-runs
```

The CLI validates the mounted samples, creates immutable input copies inside the
new run, trains one new gradient boosting pipeline, evaluates, and writes the
handoff artifacts. All five yearly inputs are required, with valid cohort fields,
labels, years, and both classes. Outputs under the source directory or original
Dataset are rejected. UUID directories, exclusive file creation, and no resume
or overwrite mode preserve existing inputs and earlier runs.

## Ingestion paths and prerequisites

See [HMDA_SOURCE.md](HMDA_SOURCE.md) for official citations, terms, vintage
limitations, and the current archive HTTP 403 blocker. Mounted mode does not
access the network or silently resample; it preserves the supplied cohort.

If you already have official CSV ZIPs named `2021_public_lar_csv.zip` through
`2025_public_lar_csv.zip`, ingest and train with:

```bash
python workflows/train_hmda.py --source archives --input-dir /path/to/archives --output-root /mnt/artifacts/hmda-training
```

For ingestion alone, append `--ingest-only`. Its run's `data/` directory can later
be supplied to `--input-dir` in mounted mode. Carry forward the original run's
`ingestion.json` and hashes as provenance; mounted mode explicitly records when
an upstream manifest is not present.

Only after confirming official-host access, network policy and adequate disk:

```bash
python workflows/train_hmda.py --source cfpb --allow-download --output-root /mnt/artifacts/hmda-training
```

This performs five potentially large downloads. It refuses redirects and
non-200 responses, limits each compressed download to 2 GiB by default, limits
CSV uncompressed size to 20 GiB per archive, and reads one CSV in chunks without
extracting archive paths. `--max-download-mb` controls the compressed cap. Allow
disk for retained archives as well as sampled outputs. An access/size/schema
failure exits nonzero, leaves diagnostic `run.json`, and never falls back to
synthetic data or a different vintage. This remote download path was tested with
controlled responses/ZIP fixtures, not live national archives.

## Training and evaluation contract

- Predictors: only the six existing fields in `backend/config.py`.
- Target: originated (1) and approved/not accepted (2) positive; denied (3) zero.
- Temporal split: 2021–2023 train, 2024 validation, 2025 holdout. Every filename,
  activity year and source year is checked before fitting.
- Shared `prepare_features()` normalizes inputs. Numeric median imputation and
  one-hot encoding are fitted on training only. Missing categoricals retain the
  `Missing` label, Exempt remains distinct, unknown categories are ignored.
  All-missing numeric columns are retained. Neither outcomes, denial reasons nor
  demographics are predictors.
- Seed 42, 200 boosting iterations, learning rate .05, 15 leaf nodes, L2=1.
  Early stopping is explicitly disabled to avoid a hidden internal random split.
  This is a documented training variant, not a promise to reproduce the existing
  model byte-for-byte. Full estimator parameters are saved.
- Threshold selected solely by 2024 balanced accuracy on .50–.99 in .01 steps;
  ties choose the lowest. Do not force the old model's .89 onto a newly fitted
  model. Test data does not select thresholds or fit preprocessing.
- Holdout metrics: ROC-AUC, accuracy, balanced accuracy, denial/positive recall,
  false-positive/negative rates, positive precision/F1, Brier score, log loss and
  confusion matrix. Serialized model predictions are checked against pre-save
  predictions. Importance uses **2024**, five permutations per feature.
- Fairness: race, ethnicity and sex group counts, actual/predicted positive rates,
  origination, error rates, recalls and group matrices. Missing demographic
  columns become an explicit unavailable group. Zero-denominator rates are null.
  Groups below 100 records, 30 denials or 30 positives are flagged; rate ranges
  use at least two sufficiently supported, named groups. This stricter reporting
  policy belongs to the new workflow and does not change the App's API contract.
- Reports are descriptive, unadjusted research, not legal findings. These public
  retrospective fields are not a verified point-in-time underwriting dataset.
  There is no borrower identifier to establish borrower-level independence.
  Because the notebook already explored 2025, a truly new validation claim needs
  a new untouched cohort. Equal annual sampling is not national volume.

## Outputs and success checks

Each run contains:

| Path | Purpose |
|---|---|
| `run.json` | Success/failure, arguments, Git SHA/dirty state, versions, split, counts, parameters, source/artifact hashes |
| `code.zip` | Exact executing workflow/helper source, dependency pins and source documentation, including uncommitted code |
| `ingestion.json`, optional `upstream_manifest.json` | Source mode/URLs, inherited provenance limitations, filtering/sampling counts and input/output hashes |
| `data/hmda_<year>.parquet` | Exact isolated input snapshots for replay |
| `models/mortgage_gradient_boosting.joblib` | New trusted full preprocessing/model pipeline |
| `models/gradient_boosting_metrics.json` | Parameters, split, selected threshold and metrics |
| `reports/gradient_boosting_evaluation_2025.json` | Holdout metrics, matrix, data/model hashes and limitations |
| `reports/threshold_selection.json` | Validation-only threshold evidence |
| `reports/fairness_2025.json` | All available demographic groups, counts, undefined rates and supported rate gaps |
| `reports/gradient_boosting_fairness_audit_2025.csv` | Compatible preferred-model race report |
| `reports/gradient_boosting_permutation_importance.csv` | Validation-only feature importance |
| `downloads/` (remote mode only) | Retained official archive bytes for exact replay |

Require exit code 0, `HMDA workflow SUCCEEDED` in Job stdout, and
`run.json.status == "succeeded"`. Review counts (default mounted: 75,000 train,
25,000 validation, 25,000 holdout), hashes, threshold selection, metric tradeoffs,
and fairness flags. A failed run leaves a `failed` manifest where the output
location was writable; partial artifacts must not be used. `run.json` does not
hash itself; it hashes the other files. Git dirty state is explicit; `code.zip`
captures the actual executed code rather than pretending HEAD includes edits.

Download/retain the run directory from Domino Job artifacts for another FDE.
Workspace `outputs/` is only local test evidence, not a durable Domino handoff.
Repeat with the same copied inputs, seed, Python and pinned packages. Timestamps,
UUIDs and provenance differ; input sampling, model predictions and metrics are
expected to agree. Cross-hardware/library changes may cause numerical differences.

## App integration and promotion boundary

The existing App continues using its original Dataset, baseline/preferred saved
models and reports. No integration change is needed to complete a training Job.
The new metrics/race/importance reports follow existing preferred-model naming,
but **do not point the live App at a new run automatically**: the App also expects
baseline artifacts not generated by this single-model workflow. Promotion requires
review of both-model compatibility, provenance, artifact completeness and a
separately approved Dataset/App update. Nothing in this command publishes,
deploys, updates registry entries or writes into `/mnt/data/HMDA_Project`.

## Verification record

See the new-workflow addendum in [AUDIT.md](../AUDIT.md) for actual commands,
results, local evidence and the remote-source limitation. Earlier audit records
refer to earlier commits; they are not proof of a new deployment.
