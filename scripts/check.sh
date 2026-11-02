#!/usr/bin/env bash
# Local/CI verification only: never trains, downloads data, or deploys.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CHECK_PYTHON="${HMDA_CHECK_PYTHON:-python}"
command -v npm >/dev/null || { echo 'Install Node/npm in the verification environment.' >&2; exit 1; }
[[ -d frontend/node_modules ]] || { echo 'Run npm ci --prefix frontend before verification.' >&2; exit 1; }
CHECK_BUILD_DIR="$(mktemp -d /tmp/hmda-check-build.XXXXXX)"
trap 'rm -rf -- "$CHECK_BUILD_DIR"' EXIT
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$CHECK_PYTHON" -m pytest -q
npm run build --prefix frontend -- --outDir "$CHECK_BUILD_DIR" --emptyOutDir
git diff --check
printf '%s\n' 'PASS: regression tests, TypeScript, production build and whitespace checks.'
