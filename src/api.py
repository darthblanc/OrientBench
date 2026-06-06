import json
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from src.runners.factory import RunnerFactory

load_dotenv()

MAX_UPLOAD_BYTES = 1 * 1024 * 1024


class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            return Response(f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit", status_code=413)
        return await call_next(request)


app = FastAPI(title="CSV Orientation Experiment")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ContentSizeLimitMiddleware)

_runs: dict[str, dict] = {}

RESULTS_DIR = Path("results")


def _execute_run(
    run_id: str, csv_path: str, model: str, id_col: str,
    n: int, seed: int, max_rows: int, cols: list[str] | None,
    api_key: str | None,
) -> None:
    try:
        runner = RunnerFactory.create(model=model, api_key=api_key)
        data = runner.run(csv_path, id_col, n=n, seed=seed, max_rows=max_rows, cols=cols)

        RESULTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{data['dataset']}_{ts}.json"
        out_path.write_text(json.dumps(data, indent=2))

        _runs[run_id] = {"status": "done", "result_path": str(out_path), "result": data}
    except Exception as exc:
        _runs[run_id] = {"status": "failed", "error": str(exc)}
    finally:
        Path(csv_path).unlink(missing_ok=True)


@app.post("/run")
async def post_run(
    csv: UploadFile = File(...),
    model: str = Form(...),
    id_col: str = Form(...),
    n: int = Form(20),
    seed: int = Form(42),
    max_rows: int = Form(15),
    cols: str = Form(""),
    api_key: str = Form(""),
) -> dict:
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
    return state


@app.get("/results/{filename}")
def get_result(filename: str) -> dict:
    p = RESULTS_DIR / filename
    if not p.exists() or p.suffix != ".json":
        raise HTTPException(status_code=404, detail="Result not found")
    return json.loads(p.read_text())


@app.get("/results")
def list_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    items = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            items.append({
                "file": p.name,
                "dataset": data.get("dataset"),
                "model": data.get("model"),
                "timestamp": data.get("timestamp"),
                "n_tasks": len(data.get("results", [])),
            })
        except Exception:
            pass
    return items
