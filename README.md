# Computer Vision POC

Object detection (YOLO) and pose (MediaPipe) on **Frigate** — same HTTP pattern as your **`Frigate and Gemini`** project (`frigate_viewer.py`, `FRIGATE_CONNECTION.md`).

## How it connects to Frigate

| What | URL |
|------|-----|
| Frames | `GET {FRIGATE_BASE_URL}/api/{camera}/latest.jpg` |
| Example | `http://192.168.100.42:5000/api/workshop/latest.jpg` |

- **No MQTT** from these scripts.
- **No direct RTSP** to the camera by default (Frigate already ingests the stream).
- Camera name = Frigate config key (`workshop`), not a Home Assistant entity.

## Setup

```powershell
cd "c:\Users\LENOVO\Desktop\Computer Vision POC"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit **`.env`**:

```ini
FRIGATE_BASE_URL=http://192.168.100.42:5000
FRIGATE_CAMERA=workshop
FRIGATE_HTTP_FPS=2
```

## Run

```powershell
python object_detection.py
python pose_detection.py
python workbench_activity.py
```

**Workbench “working” detection** (pose + YOLO, phases 2+3): explainability overlay (score bars, YOLO box colors, pose wrists). Tune `WORKBENCH_ROI` in `.env`. Pose-only: `python workbench_activity.py --no-yolo`.

Press **Q** in the OpenCV window to quit.

You should see: `Frigate HTTP poll: http://192.168.100.42:5000/api/workshop/latest.jpg`

### Test Frigate only (no YOLO/pose)

From your other project:

```powershell
cd "C:\Users\LENOVO\Desktop\Frigate and Gemini"
python frigate_viewer.py --transport http --base http://192.168.100.42:5000 --camera workshop
```

### Optional: direct RTSP / webcam

If you need to bypass Frigate:

```powershell
python object_detection.py --direct --source "rtsp://..."
```

Or clear `FRIGATE_BASE_URL` in `.env` and set `VIDEO_SOURCE=0`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `401` on RTSP | You are on **direct** mode; use Frigate HTTP (default) or fix camera password |
| `HTTP 404` on latest.jpg | Wrong `FRIGATE_CAMERA` name |
| Black / “No image” | Frigate not receiving that camera; check Frigate UI live view |
| Still using RTSP | Remove `--direct`; ensure `FRIGATE_BASE_URL` is set in `.env` |
| `mediapipe` has no attribute `solutions` | Run `pip install mediapipe==0.10.14` (newer MediaPipe removed this API) |

See also: `C:\Users\LENOVO\Desktop\Frigate and Gemini\FRIGATE_CONNECTION.md`
