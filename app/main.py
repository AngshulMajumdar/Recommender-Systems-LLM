from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.core.pipeline import run_pipeline
from app.schemas import RunConfig

APP_ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = APP_ROOT / "runs"
UPLOAD_ROOT = APP_ROOT / "uploads"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MovieLens LLM Recommender API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run_dataset(
    dataset: UploadFile = File(..., description="MovieLens 100K zip or extracted folder zipped."),
    backend: str = Form("mock"),
    model_name: str = Form("Qwen/Qwen2.5-1.5B-Instruct"),
    positive_threshold: int = Form(4),
    max_history: int = Form(10),
    max_rows: int | None = Form(None),
    top_k_json: str = Form("[5, 10, 20]"),
    use_4bit_if_available: bool = Form(True),
    save_outputs: bool = Form(True),
):
    try:
        top_k_list = json.loads(top_k_json)
        if not isinstance(top_k_list, list) or not all(isinstance(x, int) for x in top_k_list):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="top_k_json must be a JSON list of integers, e.g. [5,10,20].")

    upload_path = UPLOAD_ROOT / dataset.filename
    with open(upload_path, "wb") as f:
        while True:
            chunk = await dataset.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    config = RunConfig(
        backend=backend,
        model_name=model_name,
        positive_threshold=positive_threshold,
        max_history=max_history,
        max_rows=max_rows,
        top_k_list=top_k_list,
        use_4bit_if_available=use_4bit_if_available,
        save_outputs=save_outputs,
    )

    try:
        response = run_pipeline(str(upload_path), str(WORK_ROOT), config)
        return JSONResponse(response.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs")
def list_runs():
    runs = sorted([str(p) for p in WORK_ROOT.glob("run_*") if p.is_dir()])
    return {"runs": runs}


@app.get("/download")
def download_run(run_dir: str):
    p = Path(run_dir)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="Run directory not found.")
    zip_base = str(p)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=p)
    return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name)
