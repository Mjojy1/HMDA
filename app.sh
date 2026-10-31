#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -d .venv/bin ]]; then export PATH="$PWD/.venv/bin:$PATH"; fi
PYTHON="${HMDA_PYTHON:-python}"
export HMDA_DATA_DIR="${HMDA_DATA_DIR:-/mnt/data/HMDA_Project}"
export HMDA_REPORT_DIR="${HMDA_REPORT_DIR:-$PWD/outputs/reports}"
APP_LISTEN_PORT="${PORT:-8888}"
[[ "$APP_LISTEN_PORT" =~ ^[0-9]+$ ]] && (( APP_LISTEN_PORT > 0 && APP_LISTEN_PORT < 65536 )) || { echo 'PORT must be 1–65535'; exit 1; }
for file in backend/main.py frontend/package.json; do
  [[ -f "$file" ]] || { echo "Missing application file: $file"; exit 1; }
done
"$PYTHON" -c 'import fastapi, uvicorn, pandas, pyarrow' || { echo 'Install backend/requirements.txt in the App environment.'; exit 1; }
"$PYTHON" -m scripts.validate_artifacts --data-only
if [[ ! -f frontend/dist/index.html || "${HMDA_BUILD_FRONTEND:-0}" == 1 ]]; then
  command -v npm >/dev/null || { echo 'Node.js/npm required to build frontend. Install Node 22+ or provide a tested frontend/dist bundle.'; exit 1; }
  echo 'Building HMDA frontend'
  (cd frontend && npm ci --cache "${NPM_CONFIG_CACHE:-/tmp/hmda-npm-cache}" && npm run build)
fi
echo "Starting HMDA Mortgage Intelligence on 0.0.0.0:$APP_LISTEN_PORT"
echo 'Check api/ready in the App runtime to confirm Dataset and report availability.'
exec "$PYTHON" -m uvicorn backend.main:app --host 0.0.0.0 --port "$APP_LISTEN_PORT"
