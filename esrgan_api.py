
import os
import io
import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


ROOT = os.path.abspath(".")
RECORD_DIR = os.path.join(ROOT, "RecordFiles")
UPLOAD_DIR = os.path.join(ROOT, "uploaded_videos")
WEIGHTS_DIR = os.path.join(ROOT, "Real-ESRGAN", "weights")
ESRGAN_WEIGHT_NAME = "RealESRGAN_x4plus.pth"
ESRGAN_WEIGHT_PATH = os.path.join(WEIGHTS_DIR, ESRGAN_WEIGHT_NAME)

os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

video_caps = {}

_model_lock = threading.Lock()
_model_loaded = False
_model_obj = None
_model_type = None


def unsharp_sharpen(img_bgr: np.ndarray, amount: float = 1.0, radius: int = 1, threshold: int = 0) -> np.ndarray:
    if radius <= 0:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(img_bgr.astype(np.float32), 1.0 + amount,
                                blurred.astype(np.float32), -amount, 0.0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def encode_jpeg_bgr(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()

def load_sr_model():
    global _model_loaded, _model_obj, _model_type
    if _model_loaded:
        return
    with _model_lock:
        if _model_loaded:
            return
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            try:
                from realesrgan import RealESRGANer
                real_wrapper = RealESRGANer
            except Exception:
                from realesrgan import RealESRGAN
                real_wrapper = RealESRGAN

            if not os.path.isfile(ESRGAN_WEIGHT_PATH):
                _model_obj = None
                _model_type = "fallback"
                _model_loaded = True
                return

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            try:
                sr = real_wrapper(
                    scale=4,
                    model_path=ESRGAN_WEIGHT_PATH,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False
                )
            except TypeError:
                sr = real_wrapper(model, scale=4)
                if hasattr(sr, "load_weights"):
                    sr.load_weights(ESRGAN_WEIGHT_PATH, download=False)

            _model_obj = sr
            _model_type = "realesrgan"
            _model_loaded = True
        except Exception:
            _model_obj = None
            _model_type = "fallback"
            _model_loaded = True

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse({"error": "No filename"}, status_code=400)
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return JSONResponse({"error": "Unsupported video format"}, status_code=400)

    path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return JSONResponse({"error": f"Save failed: {e}"}, status_code=500)

    if file.filename in video_caps:
        try: video_caps[file.filename].release()
        except: pass
        del video_caps[file.filename]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return JSONResponse({"error": "Failed to open video after save"}, status_code=400)

    video_caps[file.filename] = cap
    return {"message": f"Uploaded: {file.filename}"}

@app.get("/frame/")
def get_frame(filename: str, frame_index: int = 0):
    if filename not in video_caps:
        return JSONResponse({"error": "Video not loaded. Upload first."}, status_code=400)
    cap = video_caps[filename]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ret, frame = cap.read()
    if not ret or frame is None:
        return JSONResponse({"error": f"Could not read frame {frame_index}"}, status_code=400)
    return StreamingResponse(io.BytesIO(encode_jpeg_bgr(frame)), media_type="image/jpeg")

@app.post("/zoom/")
async def zoom(
    filename: str = Form(...),
    frame_index: int = Form(...),
    center_x: int = Form(...),
    center_y: int = Form(...),
    zoom_level: int = Form(1)
):
    if not _model_loaded:
        load_sr_model()
    if filename not in video_caps:
        return JSONResponse({"error": "Video not loaded. Upload first."}, status_code=400)

    cap = video_caps[filename]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ret, frame = cap.read()
    if not ret or frame is None:
        return JSONResponse({"error": f"Could not read frame {frame_index}"}, status_code=400)

    h, w = frame.shape[:2]
    base_size = 300
    win_size = max(32, int(base_size / 1)) 
    half = win_size // 2
    cx = max(half, min(w - half, int(center_x)))
    cy = max(half, min(h - half, int(center_y)))
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    cropped = frame[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        return JSONResponse({"error": "Invalid crop"}, status_code=400)

    
    if _model_type == "realesrgan" and _model_obj:
        try:
            pil_in = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            res = _model_obj.enhance(pil_in)
            out = res[0] if isinstance(res, tuple) else res
            enhanced = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR) if isinstance(out, Image.Image) else np.array(out)
            if enhanced.dtype != np.uint8:
                enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
            return StreamingResponse(io.BytesIO(encode_jpeg_bgr(enhanced)), media_type="image/jpeg")
        except Exception:
            pass

   
    try:
        zoom_factors = {1:2, 2:3, 3:5, 4:6, 5:8}
        factor = zoom_factors.get(zoom_level, 2)
        upscaled = cv2.resize(cropped, (cropped.shape[1]*factor, cropped.shape[0]*factor), interpolation=cv2.INTER_LANCZOS4)
        sharpened = unsharp_sharpen(upscaled, amount=1.5, radius=2.0)
        return StreamingResponse(io.BytesIO(encode_jpeg_bgr(sharpened)), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": f"Fallback processing failed: {e}"}, status_code=500)


@app.get("/list_recorded")
def list_recorded():
    try:
        files = [f for f in os.listdir(RECORD_DIR) if os.path.isfile(os.path.join(RECORD_DIR, f))]
        files.sort()
        return files
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index():
    html = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Manual Zoom + ESRGAN</title>
<style>
body { font-family: Arial; padding:12px; }

</style>
</head>
<body>
<h2>Manual Zoom CCTV Viewer + SR</h2>
<div>
<input type="file" id="uploadFile" accept="video/*"/>
<button onclick="uploadVideo()">Upload</button>
&nbsp;&nbsp;
<select id="recordSel"></select>
<button onclick="useRecord()">Use Selected Record</button>
</div>

<div id="controls">
Frame Index: <input type="number" id="frameIndex" value="0" min="0" style="width:80px"/>
<button onclick="loadFrame()">Load Frame</button>
Zoom Level: 
<select id="zoomLevelSelect" onchange="changeZoom()">
<option value="1">2x</option>
<option value="2">3x</option>
<option value="3">5x</option>
<option value="4">6x</option>
<option value="5">8x</option>
</select>
<button onclick="resetZoom()">Reset</button>
</div>

<p>Click on main frame to zoom. Zoomed image appears below.</p>
<img id="frameImg" src="" alt="Frame will appear here"/>
<img id="zoomImg" src="" alt="Zoom result"/>

<script>
let filename = null;
let zoomLevel = 1;
let lastClick = {x:0,y:0};

async function refreshList() {
  try {
    const r = await fetch('/list_recorded');
    const arr = await r.json();
    const sel = document.getElementById('recordSel');
    sel.innerHTML = '';
    arr.forEach(f => { const o = document.createElement('option'); o.text=f; o.value=f; sel.appendChild(o); });
  } catch(e){ console.log(e); }
}

function useRecord() {
  const sel = document.getElementById('recordSel');
  if(!sel.value) return alert('No record selected');
  filename = sel.value;
  alert('Selected ' + filename);
}

async function uploadVideo(){
  const input = document.getElementById('uploadFile');
  if(!input.files || input.files.length===0) return alert('Choose video first');
  const file = input.files[0];
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/upload_video/', { method:'POST', body: fd });
  const j = await res.json();
  if(res.ok){ filename = file.name; alert(j.message); await refreshList(); } 
  else alert('Upload failed: ' + (j.error||JSON.stringify(j)));
}

function loadFrame(){
  if(!filename) return alert('Upload or select video first');
  const idx = parseInt(document.getElementById('frameIndex').value)||0;
  document.getElementById('frameImg').src = `/frame/?filename=${encodeURIComponent(filename)}&frame_index=${idx}&_=${Date.now()}`;
  document.getElementById('zoomImg').src = '';
}

document.getElementById('frameImg').addEventListener('click', async (e)=>{
  if(!filename) return;
  const img=e.target;
  const rect=img.getBoundingClientRect();
  const clickX=e.clientX-rect.left;
  const clickY=e.clientY-rect.top;
  const scaleX=(img.naturalWidth||img.width)/img.width;
  const scaleY=(img.naturalHeight||img.height)/img.height;
  lastClick = {x: Math.round(clickX*scaleX), y: Math.round(clickY*scaleY)};
  requestZoom(lastClick.x, lastClick.y, zoomLevel);
});

async function requestZoom(cx,cy,level){
  const idx = parseInt(document.getElementById('frameIndex').value)||0;
  const fd = new FormData();
  fd.append("filename", filename);
  fd.append("frame_index", idx);
  fd.append("center_x", cx);
  fd.append("center_y", cy);
  fd.append("zoom_level", level);
  const r = await fetch('/zoom/', { method:'POST', body: fd });
  if(!r.ok){ const jr = await r.json().catch(()=>({error:'server error'})); alert('Zoom failed: '+(jr.error||'unknown')); return; }
  const blob = await r.blob();
  document.getElementById('zoomImg').src = URL.createObjectURL(blob);
}

function changeZoom(){
  zoomLevel = parseInt(document.getElementById('zoomLevelSelect').value);
  if(lastClick.x) requestZoom(lastClick.x,lastClick.y,zoomLevel);
}
function resetZoom(){ zoomLevel=1; document.getElementById('zoomLevelSelect').value=1; if(lastClick.x) requestZoom(lastClick.x,lastClick.y,zoomLevel); }

window.onload = refreshList;
</script>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)

# ---------------- Run ----------------
if __name__ == "__main__":
    import uvicorn
    print("Starting esrgan_api app...")
    uvicorn.run("esrgan_api:app", host="0.0.0.0", port=8000, reload=True)
