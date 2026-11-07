#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HMDA_ANALYST_AGENT_URL="https://rweoriol149036.cs.domino.tech/apps-internal/6aed3aa406b33052fec8d57b"
exec ./preview.sh