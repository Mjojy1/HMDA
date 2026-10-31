#!/usr/bin/env bash
# Domino private App Preview: no Node installation or model training required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -d .venv/bin ]]; then export PATH="$PWD/.venv/bin:$PATH"; fi
PYTHON="${HMDA_PYTHON:-python}"
export HMDA_DATA_DIR="${HMDA_DATA_DIR:-/mnt/data/HMDA_Project}"
[[ -f frontend/dist/index.html ]] || { echo 'Missing versioned frontend bundle. Build frontend before publishing this commit.'; exit 1; }
if [[ -z "${HMDA_REPORT_DIR:-}" ]]; then
  export HMDA_REPORT_DIR
  HMDA_REPORT_DIR="$(mktemp -d /tmp/hmda-preview-reports.XXXXXX)"
  echo 'Evaluating saved model and generating preview reports; Dataset remains read-only.'
  "$PYTHON" -m scripts.build_reports --output "$HMDA_REPORT_DIR"
fi
"$PYTHON" -m scripts.validate_artifacts
exec ./app.sh
