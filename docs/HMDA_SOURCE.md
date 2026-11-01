# HMDA source contract and verification

Reviewed 2026-10-31 against official CFPB documentation. The workflow uses the
**Snapshot National Loan-Level Dataset**, not the dynamic or one-/three-year
revisions. Annual files are privacy-modified public Loan/Application Records.

- [Official snapshot publication](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset)
- [Official file-serving documentation](https://ffiec.cfpb.gov/documentation/api/file-serving/)
- [Public LAR fields and code definitions](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields)
- [Static dataset FAQ](https://ffiec.cfpb.gov/documentation/faq/static-dataset-faq)
- [CFPB legal notices](https://www.consumerfinance.gov/privacy/website-privacy-policy/#legal-notices)

CFPB documents `files.ffiec.cfpb.gov` as its public file host. The archive pattern
in the existing notebook and ingestion manifest is:
`https://files.ffiec.cfpb.gov/static-data/snapshot/{year}/{year}_public_lar_csv.zip`.
HEAD requests to all five 2021–2025 archive URLs returned **HTTP 403** from this
workspace; no archive was downloaded during implementation. Availability of these
specific files is therefore **not verified here**, despite their recorded prior
successful use. Do not silently substitute another release or synthetic records.
Use the mounted files now, or supply archived inputs obtained from the official
publication page. Remote execution requires egress/access to the official host.

The [2022 publication page](https://ffiec.cfpb.gov/data-publication/2022) identifies
its snapshot freeze as May 1, 2023. The official snapshot publication identifies
the 2025 snapshot as June 2, 2026; the 2024 publication identifies May 19, 2025.
Do not confuse the filing year with the release year or a later resubmission
vintage. Because URLs can outlive the bytes they originally served, every run
records input SHA-256 and, for downloads, retrieval time (run timestamps), ETag
and Last-Modified where provided. Those byte hashes, retained input copies, and
source family are the executable vintage identity. Freeze dates alone do not
prove that the mounted samples came from a particular archive.

Offline provenance: the existing mounted `ingestion_manifest_2022_2025.json`
records source URLs, archive hashes, counts, and seed 42 for 2022–2025. The workflow
copies that manifest as **historical evidence**, not an independently verified
chain from archive to sample. The 2021 source claim comes from the notebook and
has no corresponding entry in that manifest. Exact current Parquet hashes and
validated schema/cohort/labels are recorded for all five years. Reproducing the
mounted-sample run does not require recreating an unavailable upstream archive.

## Fields, filtering and use

Required fields are the six existing model features, `activity_year`, and the
cohort filters below. Optional demographic groups are `derived_race`,
`derived_ethnicity`, and `derived_sex`. Only the six features in
`backend/config.py` enter model fitting; demographic groups and outcome-related
fields are excluded from predictors.

- `action_taken` in 1, 2, 3; actions 1 and 2 map to positive, 3 to denied.
- `loan_purpose=1` (purchase), `lien_status=1` (first lien).
- `occupancy_type=1` (principal residence), `construction_method=1` (site built).
- `total_units` in 1, 2, 3, 4.
- The filing year must match the annual filename; wrong/missing years fail.

These are the notebook's cohort choices, not CFPB recommendations. Income is
reported in thousands of dollars, loan amount in dollars, and loan term in
months. Missing and Exempt values remain distinguishable in categorical inputs;
unusable numeric values become missing and are imputed using training data only.
Archive ingestion reads strings, filters all chunks, and uses a fixed-seed random
priority sample across the entire eligible stream, retaining at most 25,000
records/year. It never takes the first 25,000 rows as a substitute. Sampling
weights are not available: sample counts are not national market volumes.

CFPB's legal notices say Bureau-created information is public domain and request
source citation; third-party materials may carry separate rights. The public
HMDA publication is already modified for privacy. This implementation cites CFPB,
uses only public fields, and makes no attempt to identify applicants. Neither
public availability nor a group-rate difference establishes lawful individual
underwriting use or a legal finding. The reports are descriptive research with
missing-data, sampling, denominator, and historical-holdout limitations.
