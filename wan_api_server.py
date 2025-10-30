import os
import uuid
import base64
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import subprocess
import threading


OUTPUT_DIR = os.getenv("WAN_OUTPUT_DIR", os.path.abspath(".wan_outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="WAN 2.1 Adapter")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Public base URL for absolute links (e.g., behind reverse proxy)
PUBLIC_BASE = os.getenv("WAN_PUBLIC_BASE_URL", "").rstrip("/")

def _to_abs_url(request: Request, rel_path: str) -> str:
    if PUBLIC_BASE:
        return f"{PUBLIC_BASE}{rel_path}"
    # Fallback to request base URL
    base = str(request.base_url).rstrip("/")
    return f"{base}{rel_path}"


class T2VRequest(BaseModel):
    prompt: str
    width: Optional[int] = 1280
    height: Optional[int] = 720
    steps: Optional[int] = 30
    seed: Optional[int] = None


class FF2VRequest(BaseModel):
    prompt: str
    first_frame: str  # data URL or base64
    steps: Optional[int] = 30
    seed: Optional[int] = None


class FLF2VRequest(BaseModel):
    prompt: str
    first_frame: str
    last_frame: str
    steps: Optional[int] = 30
    seed: Optional[int] = None


def _decode_image_to_path(data_url_or_b64: str) -> str:
    if data_url_or_b64.startswith("data:"):
        b64 = data_url_or_b64.split(",", 1)[1]
    else:
        b64 = data_url_or_b64
    img_bytes = base64.b64decode(b64)
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.png")
    image.save(path)
    return path


def _run_command(cmd_template: str, **kwargs) -> str:
    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
    cmd = cmd_template.format(output=output_path, **kwargs)
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"Command failed or output not found: {cmd}")
    return output_path


# Ensure single in-flight generation to avoid GPU OOM from concurrent loads
GPU_LOCK = threading.Lock()

def _run_exclusive(func, *args, **kwargs):
    GPU_LOCK.acquire()
    try:
        return func(*args, **kwargs)
    finally:
        GPU_LOCK.release()

@app.post("/t2v")
def t2v(req: T2VRequest, request: Request):
    cmd = os.getenv("WAN_CMD_T2V")
    if not cmd:
        raise HTTPException(status_code=503, detail="WAN_CMD_T2V is not set")
    out_path = _run_exclusive(_run_command, cmd, prompt=req.prompt)
    rel = f"/outputs/{os.path.basename(out_path)}"
    return {"video_url": _to_abs_url(request, rel)}


@app.post("/ff2v")
def ff2v(req: FF2VRequest, request: Request):
    cmd = os.getenv("WAN_CMD_FF2V")
    if not cmd:
        raise HTTPException(status_code=503, detail="WAN_CMD_FF2V is not set")
    ff_path = _decode_image_to_path(req.first_frame)
    out_path = _run_exclusive(_run_command, cmd, prompt=req.prompt, first_frame=ff_path)
    rel = f"/outputs/{os.path.basename(out_path)}"
    return {"video_url": _to_abs_url(request, rel)}


@app.post("/flf2v")
def flf2v(req: FLF2VRequest, request: Request):
    cmd = os.getenv("WAN_CMD_FLF2V")
    if not cmd:
        raise HTTPException(status_code=503, detail="WAN_CMD_FLF2V is not set")
    ff_path = _decode_image_to_path(req.first_frame)
    lf_path = _decode_image_to_path(req.last_frame)
    out_path = _run_exclusive(_run_command, cmd, prompt=req.prompt, first_frame=ff_path, last_frame=lf_path)
    rel = f"/outputs/{os.path.basename(out_path)}"
    return {"video_url": _to_abs_url(request, rel)}


# Run with uv:
#   uv run --with fastapi,uvicorn -- uvicorn Wan2.1Server.wan_api_server:app --host 0.0.0.0 --port 8000
# Or with python:
#   pip install fastapi uvicorn
#   uvicorn Wan2.1Server.wan_api_server:app --host 0.0.0.0 --port 8000

