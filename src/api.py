import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from src.runners.factory import RunnerFactory

load_dotenv()

MAX_UPLOAD_BYTES = 1 * 1024 * 1024
RUN_TTL = 7200  # seconds; completed/failed runs evicted after this


class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            return Response(f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit", status_code=413)
        return await call_next(request)


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="CSV Orientation Experiment")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ContentSizeLimitMiddleware)

_runs: dict[str, dict] = {}


def _evict_stale_runs() -> None:
    cutoff = time.time() - RUN_TTL
    to_delete = [
        rid for rid, state in list(_runs.items())
        if state.get("status") != "running" and state.get("_ts", 0) < cutoff
    ]
    for rid in to_delete:
        _runs.pop(rid, None)

def _execute_run(
    run_id: str, csv_path: str, model: str, id_col: str,
    n: int, seed: int, max_rows: int, cols: list[str] | None,
    api_key: str | None,
) -> None:
    try:
        runner = RunnerFactory.create(model=model, api_key=api_key)
        data = runner.run(csv_path, id_col, n=n, seed=seed, max_rows=max_rows, cols=cols)
        _runs[run_id] = {"status": "done", "_ts": time.time(), "result": data}
    except Exception as exc:
        _runs[run_id] = {"status": "failed", "_ts": time.time(), "error": str(exc)}
    finally:
        Path(csv_path).unlink(missing_ok=True)


@app.post("/run")
@limiter.limit("5/minute")
async def post_run(
    request: Request,
    csv: UploadFile = File(...),
    model: str = Form(...),
    id_col: str = Form(...),
    n: int = Form(20),
    seed: int = Form(42),
    max_rows: int = Form(15),
    cols: str = Form(""),
    api_key: str = Form(""),
) -> dict:
    _evict_stale_runs()
    run_id = str(uuid.uuid4())
    cols_list = [c.strip() for c in cols.split(",") if c.strip()] if cols else None
    key = api_key.strip() or None

    suffix = Path(csv.filename).suffix if csv.filename else ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await csv.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
        tmp.write(data)
        tmp_path = tmp.name

    _runs[run_id] = {"status": "running"}

    threading.Thread(
        target=_execute_run,
        args=(run_id, tmp_path, model, id_col, n, seed, max_rows, cols_list, key),
        daemon=True,
    ).start()

    return {"id": run_id, "status": "running"}


@app.get("/run/{run_id}")
def get_run(run_id: str) -> dict:
    state = _runs.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    response = {k: v for k, v in state.items() if not k.startswith("_")}
    if state.get("status") in ("done", "failed"):
        _runs.pop(run_id, None)
    return response


