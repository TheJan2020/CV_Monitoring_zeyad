# Frigate connection (this POC)

This repo follows the same rules as **`C:\Users\LENOVO\Desktop\Frigate and Gemini\FRIGATE_CONNECTION.md`**.

## Default: HTTP `latest.jpg`

```text
GET {FRIGATE_BASE_URL}/api/{FRIGATE_CAMERA}/latest.jpg
```

Set in `.env`:

```ini
FRIGATE_BASE_URL=http://192.168.100.42:5000
FRIGATE_CAMERA=workshop
```

Run:

```powershell
python object_detection.py
python pose_detection.py
```

Implementation: `frigate_http.py` (urllib poll, same idea as `frigate_viewer.py --transport http`).

## Not used here

- Frigate **MQTT** (broker in Frigate’s own YAML only).
- Direct **RTSP** to `192.168.100.37` unless you pass `--direct` and `VIDEO_SOURCE`.
- Frigate **`/ws`** JSON socket (UI state only).
- **`/live/jsmpeg/{camera}`** WebSocket (available in `frigate_viewer.py --transport jsmpeg` in the other project).

## Live WebSocket (optional, other repo)

For smooth live video like Frigate’s Live UI:

```powershell
python frigate_viewer.py --transport jsmpeg --base http://192.168.100.42:5000 --camera workshop
```

This POC uses HTTP snapshots for simplicity (works on Windows without extra ffmpeg setup).
