# Integration and deployment readiness audit

Audit scope: existing HMDA Mortgage Intelligence, real files in `/mnt/data/HMDA_Project`, and the existing grant-based Domino draft preview. No training, HMDA regeneration, Dataset writes, Git commits/pushes, publication, or deployment changes were performed during this audit.

## Result

- Local fixed code: **53/53 independent real-data API checks pass**.
- Existing deployed commit `793c6ec`: **51/53 pass**. Its two failures are categorical missing-value counts on the lineage page; these are fixed locally.
- All five pages render in Chromium against both local and deployed APIs. Desktop/mobile, filters, navigation, console errors and failed HTTP responses checked. No console errors or failed responses in either run.
- TypeScript and Vite production build pass; a 572 kB / 165 kB gzip JavaScript bundle-size advisory remains.
- Backend regression suite: **25 tests pass**. Fixtures are isolated from production files.

## Real-data verification

Independent Parquet aggregation (125,000 total, 25,000 per year):

| Outcome | Count | Share |
|---|---:|---:|
| Originated, action 1 | 110,155 | 88.1240% |
| Approved but not accepted, action 2 | 3,596 | 2.8768% |
| Denied, action 3 | 11,249 | 8.9992% |
| Saved positive target, actions 1 + 2 | 113,751 | 91.0008% |

All overview, annual trend, histogram totals/missing counts, and filter checks matched. Equal annual counts describe sampling, not national application volumes.

Model comparison values match `models/model_metrics.json` and `models/gradient_boosting_metrics.json`. Preferred 2025 metrics: ROC-AUC 0.7784548518, accuracy 0.88208, balanced accuracy 0.7071016146, denial recall 0.4951812758; threshold 0.89. Importance values match **only** `reports/gradient_boosting_permutation_importance.csv`.

The preferred fairness CSV is still absent from the Dataset itself. The existing preview uses its generated `gradient_boosting_fairness_audit_2025.csv` in a temporary preview report directory, derived from the existing Dataset and saved preferred model. This is not a baseline fallback or mock. The local app uses the corresponding existing report in `outputs/reports`.

For this audit, the saved preferred pipeline was evaluated in memory, without fitting or writing reports. Every displayed group's application count, actual denials, origination rate, historical positive rate, predicted positive rate and denial recall matched. The confusion matrix matched `[[1079,1100],[1848,20973]]`; model/data hashes matched. Zero-denial groups retain null recall; small groups are flagged and filtered; unavailable demographics are retained. Comparisons remain descriptive and unadjusted.

## Audited fixes

1. **Incorrect lineage counts:** categorical nulls had been normalized to `Missing` before `.isna()` was counted. The deployed app reports zero for 1,253 missing debt-to-income values and 313 missing loan terms. Local API/UI now correctly reports these and separately reports Exempt counts.
2. **Hardcoded model context:** the fairness threshold and several evaluation-year/split labels are now driven by saved metadata responses. Chart statistics and comparison scores were already API-backed; no mock dashboard data was found.
3. **Insufficient report validation:** reject duplicate/unrecognized groups, invalid/fractional counts, incorrect group denominators/rates and inappropriate null recall. Evaluation readiness now checks matrix consistency, saved metrics/threshold and model/test-data hashes.
4. **API docs behind proxy:** Swagger now fetches a relative OpenAPI URL under `api/`, instead of bypassing the Domino prefix.
5. **Demographic warnings:** small-group and missing-demographic warnings can appear together instead of one hiding the other. Removed stale governance copy asserting no App deployment exists.
6. Expanded regression tests, reusable read-only API audit, browser console/network checking, and rebuilt the versioned frontend bundle.

Source mapping: Executive Overview → overview/trends/metrics; Lending → filters/overview/trends/distributions; Model → metrics/importance/evaluation; Fair-Lending → fairness (preferred report plus metadata); Governance → lineage/metrics/ready. Backend source files determine the results. Static explanatory text, known HMDA code labels, histogram bin boundaries and sample-size cutoffs are not simulated data.

## Domino verification and next steps

Existing preview: `https://rweoriol149036.cs.domino.tech/preview/6ae541e906b33052fec8c17d/`

At audit time, read-only management checks confirmed grant-based visibility, draft status, `Running`, mounted Datasets, commit `793c6ecb125c199e4e31dd856d790750fcb75f8c`, and `preview.sh` as version entry point. Logs show `preview.sh` running its saved-model evaluation and artifact validation, then executing `app.sh` on `0.0.0.0:8888`. Authenticated dashboard/API and readiness checks pass. The locally fixed `app.sh` was also started separately on port 8003 for audit testing.

Publication preparation reviewed the complete change set from `origin/main`: no credential literals, Dataset/model files, local reports, logs, or files over 1 MB were found. The production bundle is intentionally versioned for a Node-free preview; rebuilding it produced identical tracked assets. Backend regression tests passed again (25 tests). The audited implementation and fixes were already committed as `793c6ec` and `1c1ab69`; preserve those commits when advancing `main`.

After the approved push to `origin/main`, update the private preview manually:

1. In the `HMDA_Project` Workspace, use the Git branch selector to select `main` and pull the latest remote commit if needed. Keep automatic preview synchronization off until ready to review.
2. Open **App preview**, then **Preview settings**. Verify **Git reference** identifies `main`. This display follows the workspace branch; it is not a separate editable branch selector.
3. When **New commits available** appears, verify **Latest commit** against the SHA reported after the push and click **Update to latest**. Wait for syncing/rebuilding to finish. **Save and reload** is for settings changes; no entry-point or environment change is required here. Keep `preview.sh`, the existing Compute Environment, port 8888, Dataset mounting, and restricted access.
4. Open the preview URL plus `api/ready`: require HTTP 200, `status: ready`, and `ready` for data, metrics, importance, fairness, and evaluation. Confirm `api/lineage` reports the new Git SHA.
5. On **Data lineage & governance**, confirm missing/nonnumeric counts: debt-to-income **1,253**, loan term **313**. Both separately show **2,325 exempt**.
6. Review all five pages: **Executive overview** (125,000 records); **Lending analytics** (filters change records/charts); **Model intelligence** (gradient boosting ROC-AUC 0.7785, threshold 0.89, importance and confusion matrix); **Fair-lending monitoring** (size filter, limited-support and unavailable-demographic warnings); **Data lineage & governance** (commit, sources, corrected counts, readiness). Check browser Console for errors and Network for failed API requests.
7. Stop at the private preview. Do not publish or deploy until separately approved.

The existing `preview.sh` evaluates the saved model into temporary reports when starting a fresh runtime; it does not train or modify Dataset files. Neither entry-point script needs changing. Workspace `outputs/` is not copied to new App containers.

## Reproduce checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PATH="/mnt/code/.venv/bin:$PATH" npm run build --prefix frontend
PORT=8003 ./app.sh
.venv/bin/python -m scripts.audit_integration --url http://127.0.0.1:8003/ --output outputs/audit/local.json
PLAYWRIGHT_BROWSERS_PATH=/tmp/hmda-browsers HMDA_TEST_URL=http://127.0.0.1:8003/ .venv/bin/python -m scripts.browser_smoke
```

Private preview checks support `--domino-auth` for `audit_integration`, and `HMDA_TEST_DOMINO_AUTH=1` with the preview URL for `browser_smoke`. Temporary workspace tokens remain in memory and are never written to source or reports. Evidence is in ignored `outputs/audit/`.
