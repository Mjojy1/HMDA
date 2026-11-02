# Production-readiness evidence

Scope: a historical HMDA analytics and research application in Domino, not an
individual lending-decision service or a certification of legal compliance.

## Engineering requirements

| Requirement | Evidence | Status |
|---|---|---|
| Domino project | Existing HMDA_Project; owner reports published App and completed training Job | Project exists; current deployment/Job execution not re-audited for this slide session |
| Modular code | `backend/` API/data services, `frontend/` UI, `workflows/` input/training, `jobs/` saved-model evaluation, `experiments/` tracking | Implemented |
| Version control | Core workflow committed to `main`, commit `9c280f18779aa4dbbb2d98afea363faf5a09d274` | Verified locally; new session deliverables remain uncommitted pending approval |
| Automated verification | `scripts/check.sh`: regression tests, TypeScript and isolated production build | Run locally; not a claim of hosted CI or branch protection |
| Reproducibility | Fixed seeds, explicit temporal split, train-only preprocessing, package versions, data/model/code hashes and code snapshot | Implemented and tested |
| Input safety | Schema, year, cohort and target validation; offline Parquet default; no synthetic fallback | Implemented and tested |
| Artifact isolation | Unique run directories, exclusive writes, original Dataset/model preservation | Implemented and tested |
| Runtime readiness | `/api/health`, `/api/ready`, fail-closed validation of required reports | Implemented; use deployed URL for current runtime checks |
| Controlled model promotion | New training outputs are candidates; app retains its existing artifact set | Preserved; promotion remains an explicit review/deployment action |

## One verification command

In a verification environment with Python requirements and the frontend lockfile
installed (`pip install -r workflows/requirements.txt -r tests/requirements.txt`
and `npm ci --prefix frontend`), run:

```bash
bash scripts/check.sh
```

In the current Workspace, use the existing local tools:

```bash
PATH="/mnt/code/.venv/bin:$PATH" HMDA_CHECK_PYTHON=.venv/bin/python bash scripts/check.sh
```

The script writes frontend build output to a unique temporary directory and
cleans only that directory. It does not change the versioned app bundle, train
models, access the Dataset, submit Jobs or deploy. The regression fixtures are
synthetic test inputs, never dashboard substitutes. No additional hosted CI
workflow has been enabled or claimed.

## What “production-ready” does and does not mean here

The project has production-oriented structure, validation, reproducibility,
version control and automated tests appropriate to this exercise. It is not
proven ready for arbitrary traffic, regulated automated underwriting or an
unattended model-promotion process. The remaining operational acceptance checks
are explicit:

- Record the actual completed training Job ID, artifact location and deployed
  App version/environment. User confirmation alone is not log-level verification.
- Configure and verify production availability/error monitoring, alert owners,
  retention/backup, rollback and incident procedures.
- Load/resource-test the expected concurrency and runtime footprint; define an
  agreed service objective. No load test was performed in this work.
- Review dependency vulnerabilities and organization-specific access controls;
  a successful build is not a security assessment.
- Add hosted CI/branch protection if required by the delivery team; do not imply
  that a local check script enforces repository policy.
- Review model/data suitability and a new untouched validation cohort before
  high-stakes use. Historical group-rate differences are descriptive, not legal
  findings. Public fields do not establish point-in-time feature availability.

None of these requires replacing the existing serving model merely because a
new reproducible training workflow exists. See [HANDOFF.md](HANDOFF.md) for the
Job command, artifacts and promotion boundary.
