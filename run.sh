#!/usr/bin/env bash
# This file starts a search using the project virtual environment if available.
# Example: ./run.sh 8 2 runs class (8, 2).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
if [ -x "venv/bin/python3" ]; then
    PY="venv/bin/python3"
else
    PY="${PYTHON:-python3}"
fi
exec "$PY" run.py "$@"
