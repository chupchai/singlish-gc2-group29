#!/usr/bin/env bash
set -euo pipefail
mvp_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$mvp_root/.."
exec "$mvp_root/.venv/bin/python" -m uvicorn mvp.app:app --host 127.0.0.1 --port "${PORT:-8000}"
