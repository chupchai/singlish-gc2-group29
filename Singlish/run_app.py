"""
CML Application launcher for MH's Singlish web app (mvp.app:app).

CML runs this script INSIDE a Jupyter kernel and monitors that kernel. So we
must NOT replace/kill the process (os.exec*) -- doing so makes CML report
"The Jupyter kernel stopped. Will stop the workload." and exit.

The kernel already owns a running asyncio event loop, which normally makes
uvicorn.run() fail with "asyncio.run() cannot be called from a running event
loop". nest_asyncio.apply() patches that loop to allow the reentrant call, so
uvicorn can start and serve inside the kernel while keeping it alive.
"""
import os
import sys

# --- Locate the repo root (the folder that contains the `mvp/` package) ---
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is undefined when run inside a notebook/kernel cell.
    _here = os.getcwd()

REPO_ROOT = os.environ.get("SINGLISH_REPO", _here)

# If we didn't land on the repo, fall back to the known Cloudera path.
if not os.path.isdir(os.path.join(REPO_ROOT, "mvp")):
    REPO_ROOT = "/home/cdsw/Singlish"

if not os.path.isdir(os.path.join(REPO_ROOT, "mvp")):
    raise SystemExit(
        f"'mvp' package not found in {REPO_ROOT!r}. "
        "Place run_app.py in the repo root or set SINGLISH_REPO."
    )

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# --- Start uvicorn the right way for the launch context ---
#
# CML runs this inside a Jupyter kernel and executes the script as "startup
# chunks" that are expected to RETURN. uvicorn.run() blocks forever, so the
# launcher times it out and reports "Finish running startup chunks: failed".
#
# Fix: if a running event loop exists (the kernel case), schedule the server as
# a background task so the chunk returns immediately while the kernel keeps the
# loop -- and therefore the server -- alive. In a plain terminal/subprocess
# there is no running loop, so we just block and serve normally.
import asyncio
import uvicorn

port = int(os.environ.get("CDSW_APP_PORT", "8098"))
config = uvicorn.Config("mvp.app:app", host="127.0.0.1", port=port)
server = uvicorn.Server(config)

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None

if loop is not None and loop.is_running():
    # CML Jupyter kernel: non-blocking, keep the startup chunk short.
    loop.create_task(server.serve())
else:
    # Terminal / plain subprocess: block and serve.
    server.run()
