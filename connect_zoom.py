import time
import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from onvif import ONVIFCamera

# =========================
# Configuration
# =========================
MODEL_PATH = "new_best.pt"
CAMERA_IP = ""
PORT = 80
USERNAME = ""
PASSWORD = ""

CONF_THRESH = 0.25
ZOOM_IN_DURATION = 1.6     # seconds to apply zoom-in velocity
ZOOM_HOLD_SEC   = 3.0      # hold time at zoomed-in view
ZOOM_OUT_DURATION = 1.6
CENTER_EPS = 0.05
RETURN_EPS = 0.02
RETURN_TIMEOUT = 10.0
TEMP_PRESET_NAME = "RETURN_VIEW_TEMP"

SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# YOLO
# =========================
model = YOLO(MODEL_PATH)
try:
    model.to("cuda")
except Exception:
    print("⚠️ Running YOLO on CPU")

# =========================
# ONVIF Setup
# =========================
cam = ONVIFCamera(CAMERA_IP, PORT, USERNAME, PASSWORD)
media = cam.create_media_service()
ptz = cam.create_ptz_service()
profile = media.GetProfiles()[0]

# =========================
# PTZ helpers
# =========================
def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

def get_status_position():
    st = ptz.GetStatus({'ProfileToken': profile.token})
    return st.Position

def clone_vector(vec):
    new = type(vec)()
    if hasattr(vec, "PanTilt") and vec.PanTilt is not None:
        pt = type(vec.PanTilt)()
        pt.x = vec.PanTilt.x
        pt.y = vec.PanTilt.y
        new.PanTilt = pt
    if hasattr(vec, "Zoom") and vec.Zoom is not None:
        zm = type(vec.Zoom)()
        zm.x = vec.Zoom.x
        new.Zoom = zm
    return new

def continuous_move(pan=0.0, tilt=0.0, zoom=0.0, duration=0.4):
    req = ptz.create_type('ContinuousMove')
    req.ProfileToken = profile.token
    st = ptz.GetStatus({'ProfileToken': profile.token})
    req.Velocity = st.Position
    if pan != 0.0 or tilt != 0.0:
        req.Velocity.PanTilt.x = clamp(pan)
        req.Velocity.PanTilt.y = clamp(tilt)
    if zoom != 0.0:
        req.Velocity.Zoom.x = clamp(zoom)
    ptz.ContinuousMove(req)
    time.sleep(duration)
    ptz.Stop({'ProfileToken': profile.token})

def zoom_in(duration=ZOOM_IN_DURATION):
    continuous_move(zoom=0.5, duration=duration)

def zoom_out(duration=ZOOM_OUT_DURATION):
    continuous_move(zoom=-0.5, duration=duration)

def center_on(cx, cy, max_steps=8, step_time=0.35):
    for _ in range(max_steps):
        off_x = cx - 0.5
        off_y = cy - 0.5
        if abs(off_x) < CENTER_EPS and abs(off_y) < CENTER_EPS:
            break
        pan = clamp(off_x * 2.0)
        tilt = clamp(-off_y * 2.0)
        continuous_move(pan=pan, tilt=tilt, duration=step_time)

def absolute_move_to(pos):
    req = ptz.create_type('AbsoluteMove')
    req.ProfileToken = profile.token
    req.Position = pos
    try:
        sp = ptz.GetStatus({'ProfileToken': profile.token}).Position
        req.Speed = sp
    except Exception:
        pass
    ptz.AbsoluteMove(req)

def wait_until_reached(target_pos, eps=RETURN_EPS, timeout=RETURN_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        try:
            cur = get_status_position()
            dx = abs((cur.PanTilt.x if cur.PanTilt else 0) - (target_pos.PanTilt.x if target_pos.PanTilt else 0))
            dy = abs((cur.PanTilt.y if cur.PanTilt else 0) - (target_pos.PanTilt.y if target_pos.PanTilt else 0))
            dz = abs((cur.Zoom.x if cur.Zoom else 0) - (target_pos.Zoom.x if target_pos.Zoom else 0))
            if dx < eps and dy < eps and dz < eps:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False

def set_or_update_temp_preset(name=TEMP_PRESET_NAME):
    try:
        presets = ptz.GetPresets({'ProfileToken': profile.token}) or []
        token = None
        for p in presets:
            if getattr(p, 'Name', None) == name:
                token = p.token
                break
        if token:
            token = ptz.SetPreset({'ProfileToken': profile.token, 'PresetToken': token, 'PresetName': name})
            return token
        else:
            token = ptz.SetPreset({'ProfileToken': profile.token, 'PresetName': name})
            return token
    except Exception as e:
        print(f" Preset set/update failed: {e}")
        return None

def goto_preset(token):
    req = ptz.create_type('GotoPreset')
    req.ProfileToken = profile.token
    req.PresetToken = token
    try:
        sp = ptz.GetStatus({'ProfileToken': profile.token}).Position
        req.Speed = sp
    except Exception:
        pass
    ptz.GotoPreset(req)

def return_to_saved_view(saved_pos, preset_token=None):
    if preset_token:
        try:
            goto_preset(preset_token)
            if wait_until_reached(saved_pos):
                return True
        except Exception as e:
            print(f" GotoPreset failed: {e}")

    try:
        absolute_move_to(saved_pos)
        if wait_until_reached(saved_pos):
            return True
    except Exception as e:
        print(f" AbsoluteMove fallback failed: {e}")
    return False

# =========================
# Open RTSP
# =========================
rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/stream"
cap = cv2.VideoCapture(rtsp_url)

cycle_index = 0

print("▶ Detect → Zoom one → Return → Save crop → Repeat")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Lost RTSP, reconnecting...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(rtsp_url)
        continue

    results = model(frame, conf=CONF_THRESH)
    boxes = results[0].boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        def area(b):
            x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy())
            return (x2 - x1) * (y2 - y1)
        sorted_boxes = sorted(list(boxes), key=area, reverse=True)

        for b in sorted_boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            cx = (x1 + x2) / 2 / frame.shape[1]
            cy = (y1 + y2) / 2 / frame.shape[0]
            detections.append((cx, cy, (x1, y1, x2, y2)))

    annotated = results[0].plot()
    cv2.imshow("PTZ Detect→Zoom cycle", annotated)

    if detections:
        pick_idx = cycle_index % len(detections)
        cx, cy, (x1, y1, x2, y2) = detections[pick_idx]

        # Save cropped image
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"obj_{cycle_index}_{timestamp}.jpg")
            cv2.imwrite(filename, crop)
            print(f" Saved crop: {filename}")

        saved_pos = clone_vector(get_status_position())
        preset_token = set_or_update_temp_preset(TEMP_PRESET_NAME)

        print(f"\n Cycle #{cycle_index+1}: object {pick_idx+1}/{len(detections)} | center=({cx:.2f},{cy:.2f})")

        center_on(cx, cy)
        zoom_in(ZOOM_IN_DURATION)
        time.sleep(ZOOM_HOLD_SEC)
        zoom_out(ZOOM_OUT_DURATION)

        ok = return_to_saved_view(saved_pos, preset_token)
        if not ok:
            print(" Return to previous position timed out/failed. Continuing...")

        cycle_index += 1
    else:
        time.sleep(0.05)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
