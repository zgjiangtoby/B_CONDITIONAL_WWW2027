#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" "$(dirname "$0")/main.py" "$@"
