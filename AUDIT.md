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

## Local fixes awaiting approval

1. **Incorrect lineage counts:** categorical nulls had been normalized to `Missing` before `.isna()` was counted. The deployed app reports zero for 1,253 missing debt-to-income values and 313 missing loan terms. Local API/UI now correctly reports these and separately reports Exempt counts.
2. **Hardcoded model context:** the fairness threshold and several evaluation-year/split labels are now driven by saved metadata responses. Chart statistics and comparison scores were already API-backed; no mock dashboard data was found.
3. **Insufficient report validation:** reject duplicate/unrecognized groups, invalid/fractional counts, incorrect group denominators/rates and inappropriate null recall. Evaluation readiness now checks matrix consistency, saved metrics/threshold and model/test-data hashes.
4. **API docs behind proxy:** Swagger now fetches a relative OpenAPI URL under `api/`, instead of bypassing the Domino prefix.
5. **Demographic warnings:** small-group and missing-demographic warnings can appear together instead of one hiding the other. Removed stale governance copy asserting no App deployment exists.
6. Expanded regression tests, reusable read-only API audit, browser console/network checking, and rebuilt the versioned frontend bundle.

Source mapping: Executive Overview → overview/trends/metrics; Lending → filters/overview/trends/distributions; Model → metrics/importance/evaluation; Fair-Lending → fairness (preferred report plus metadata); Governance → lineage/metrics/ready. Backend source files determine the results. Static explanatory text, known HMDA code labels, histogram bin boundaries and sample-size cutoffs are not simulated data.

## Domino verification and next steps

Existing preview: `https://rweoriol149036.cs.domino.tech/preview/6ae541e906b33052fec8c17d/`

Read-only management checks confirm grant-based visibility, draft status, `Running`, mounted Datasets, commit `793c6ecb125c199e4e31dd856d790750fcb75f8c`, and `preview.sh` as version entry point. Logs show `preview.sh` running its saved-model evaluation and artifact validation, then executing `app.sh` on `0.0.0.0:8888`. Authenticated dashboard/API and readiness checks pass. The locally fixed `app.sh` was also started separately on port 8003 for audit testing.

No configuration change is required just to view the existing preview. To apply fixes:

1. Review and approve the local changes, including the rebuilt `frontend/dist` bundle, before committing and pushing the preview branch.
2. After that approval and push, open **App Preview** in `HMDA_Project` and reload/sync it to the new commit (the preview may auto-sync pushed commits). Keep launch script **`preview.sh`**, the existing Python Compute Environment, port **8888**, Dataset mounting enabled, and restricted access.
3. Confirm `api/ready` is HTTP 200, all five pages load, and lineage shows the corrected missing counts. Verify the new commit in lineage/runtime metadata.
4. Only if separately approved, publish a version and deploy it. Do not publish the current preview as though it contains the uncommitted audit fixes.

If selecting `app.sh` directly instead of `preview.sh`, provide the existing preferred reports via a durable `HMDA_REPORT_DIR`. A fresh container does not inherit workspace `outputs/`. `app.sh` intentionally does not generate reports; `preview.sh` generates only analytical reports from existing trusted inputs. No model training or data regeneration is needed.

## Reproduce checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PATH="/mnt/code/.venv/bin:$PATH" npm run build --prefix frontend
PORT=8003 ./app.sh
.venv/bin/python -m scripts.audit_integration --url http://127.0.0.1:8003/ --output outputs/audit/local.json
PLAYWRIGHT_BROWSERS_PATH=/tmp/hmda-browsers HMDA_TEST_URL=http://127.0.0.1:8003/ .venv/bin/python -m scripts.browser_smoke
```

Private preview checks support `--domino-auth` for `audit_integration`, and `HMDA_TEST_DOMINO_AUTH=1` with the preview URL for `browser_smoke`. Temporary workspace tokens remain in memory and are never written to source or reports. Evidence is in ignored `outputs/audit/`.
