#!/usr/bin/env python3
"""
Document AI Extractor — Web Backend
------------------------------------
Wraps the HybridExtractor (YOLO + Qwen2-VL) from executable.py behind a
simple HTTP API, and serves the frontend.

Real mode:
    If executable.py + its model dependencies (torch, transformers, ultralytics,
    qwen_vl_utils, fitz/PyMuPDF) are importable AND a GPU/CPU can actually load
    the models, every upload is run through your real HybridExtractor.

Demo mode (automatic fallback):
    If the real model stack isn't available (e.g. no GPU, models not downloaded
    yet, missing packages), the server automatically falls back to replaying
    real historical results from sample_results.json so you can build and test
    the UI immediately. The response includes "demo_mode": true in that case
    so the frontend can show a small badge.

Run:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000 in your browser.
"""

import io
import json
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
SAMPLE_RESULTS_PATH = BASE_DIR / "sample_results.json"

app = FastAPI(title="Document AI Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Try to load the real extractor. This is expensive (loads Qwen2-VL + YOLO
# onto the GPU/CPU) so we do it once, lazily, on first real request — not at
# import time — and cache whether it succeeded.
# --------------------------------------------------------------------------
_extractor = None
_extractor_load_failed = False
_extractor_error = None


def get_extractor():
    """Lazily construct the real HybridExtractor. Returns None on failure."""
    global _extractor, _extractor_load_failed, _extractor_error

    if _extractor is not None:
        return _extractor
    if _extractor_load_failed:
        return None

    try:
        # executable.py must be importable (same repo). Add its folder to path.
        sys.path.insert(0, str(BASE_DIR))
        from executable import HybridExtractor  # noqa: E402

        _extractor = HybridExtractor()
        return _extractor
    except Exception as exc:  # noqa: BLE001 - we want to fall back on ANY failure
        _extractor_load_failed = True
        _extractor_error = f"{type(exc).__name__}: {exc}"
        print("⚠️  Real model stack unavailable, falling back to demo mode.")
        print(f"    Reason: {_extractor_error}")
        return None


# --------------------------------------------------------------------------
# Demo-mode data: replay real historical extraction results so the UI has
# something honest to show before the real model pipeline is wired up.
# --------------------------------------------------------------------------
_demo_results = None


def get_demo_results():
    global _demo_results
    if _demo_results is None:
        with open(SAMPLE_RESULTS_PATH, "r", encoding="utf-8") as f:
            _demo_results = json.load(f)
    return _demo_results


def build_demo_response(filename: str) -> dict:
    samples = get_demo_results()
    base = random.choice(samples)
    # Simulate a bit of processing time so the UI's loading state feels real.
    time.sleep(random.uniform(0.6, 1.4))
    result = json.loads(json.dumps(base))  # deep copy
    result["doc_id"] = Path(filename).stem or result.get("doc_id", "document")
    result["demo_mode"] = True
    return result


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "real_model_loaded": _extractor is not None,
        "real_model_load_failed": _extractor_load_failed,
        "error": _extractor_error,
    }


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    allowed_suffixes = {".pdf", ".png", ".jpg", ".jpeg"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use PDF, PNG, or JPG.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    extractor = get_extractor()

    if extractor is None:
        # Demo mode fallback
        return build_demo_response(file.filename)

    # Real mode: write to a temp file (the extractor reads from a path) and run it.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = extractor.process(tmp_path)
        result["doc_id"] = Path(file.filename).stem
        result["demo_mode"] = False
        return result
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Serve the frontend
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)