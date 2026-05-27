# Running on the Windows RTX 5000 box

This guide moves the pipeline to your Windows GPU machine and exposes the live dashboard at `http://<gpu-box-ip>:8000`.

---

## 0. (Recommended) Enable SSH so you can drive the box from your Mac

Windows 10/11 ship with a built-in OpenSSH server. Enable it once at the Windows keyboard; after that everything below can be done from a terminal on your Mac. **Skip this section if you prefer to do the setup at the Windows keyboard.**

### 0a. On the Windows machine — one-time setup

Open **PowerShell as Administrator** and paste:

```powershell
# Enable OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start it now and on every boot
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

# Firewall rule (usually auto-added; this is a safety net)
if (!(Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
}

# Print username + LAN IP
whoami
ipconfig | findstr IPv4
```

Note the username (e.g. `DESKTOP-ABC\zeyad`) and IPv4 address (e.g. `192.168.100.50`).

### 0b. On your Mac — connect

```bash
ssh zeyad@192.168.100.50
```

Enter the Windows password — you'll land in a PowerShell prompt on the GPU box.

### 0c. (Optional but recommended) Passwordless key-based auth

Once basic SSH works, do this once from your Mac so you never type the Windows password again:

```bash
# Generate a key if you don't have one
test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N ""

# Install your public key on the Windows box
ssh zeyad@192.168.100.50 "powershell -Command \"if (!(Test-Path \$HOME\.ssh)) { New-Item -Type Directory -Path \$HOME\.ssh }; Add-Content -Path \$HOME\.ssh\authorized_keys -Value '$(cat ~/.ssh/id_ed25519.pub)'\""
```

After that, `ssh zeyad@192.168.100.50` logs in instantly.

> SSH gives you a **terminal only** — you won't see the OpenCV window over SSH. That's exactly why this project has the web dashboard (`--web`). For full graphical desktop access use Windows **Remote Desktop (RDP)** (built-in on Windows Pro) or **TeamViewer** / **AnyDesk**.

---

## 1. Prerequisites (on the Windows machine)

Open **PowerShell** (right-click → Run as Administrator for the first install):

```powershell
# 1a. Check NVIDIA driver is installed and the GPU is visible
nvidia-smi
# You should see "Quadro RTX 5000" / "RTX 5000 Ada" with driver version + CUDA version (e.g. 12.x)

# 1b. Install Python 3.11 (skip if already installed and `python --version` >= 3.10)
winget install -e --id Python.Python.3.11

# 1c. Install Git (skip if `git --version` works)
winget install -e --id Git.Git

# Close and reopen PowerShell so the PATH refreshes.
```

## 2. Clone the repo

```powershell
cd $HOME\Desktop
git clone https://github.com/BaselYAS/CV_Monitoring.git
cd CV_Monitoring
```

## 3. Create venv + install with CUDA PyTorch

The default `pip install torch` on Windows pulls the **CPU** wheel — we need the CUDA one. The right index URL depends on your driver's CUDA version (shown in `nvidia-smi` top-right). For driver supporting CUDA 12.x, use `cu124`. For CUDA 11.8, use `cu118`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA-enabled PyTorch FIRST (this is the critical step)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then the rest of the requirements
pip install -r requirements.txt
```

Verify CUDA works:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```
Expected output: `cuda: True | device: NVIDIA RTX 5000 ...`

If you see `cuda: False`, the CPU wheel got installed. Fix:
```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 4. Configure `.env`

The `.env` file from the Mac already points at the RTSP camera. On Windows, copy it over (or edit fresh):

```ini
# Direct RTSP — sub stream (low res, fast). Switch _sub → _main for full HD.
VIDEO_SOURCE=rtsp://USERNAME:PASSWORD@<CAMERA_IP>:554/Preview_01_sub

# Force YOLO onto the RTX 5000
YOLO_DEVICE=cuda:0

# Quieter terminal at high FPS
LOG_VERBOSE=0
```

(All other variables can stay at their defaults.)

## 5. Run with the web dashboard

```powershell
python workbench_activity.py --web
```

Startup output should look like:
```
Direct video source: 'rtsp://USERNAME:PASSWORD@<CAMERA_IP>:554/Preview_01_sub'
YOLO device: cuda:0
ROI: full frame
Pipeline: people (YOLO-pose) | devices [phone, laptop, keyboard, mouse] | extra COCO tools
web dashboard: http://0.0.0.0:8000/  (open from any device on LAN)
```

## 6. Open the dashboard

Find the Windows box's LAN IP:
```powershell
ipconfig | findstr IPv4
```
e.g. `IPv4 Address. . . . . . . . . . . : 192.168.100.50`

From your **Mac** (or any device on the same network), open:
```
http://192.168.100.50:8000/
```

You'll see the live annotated MJPEG feed on the left and a detection panel on the right showing:
- Current activity state (IDLE / PRESENT / WORKING / PHONE)
- Live medium and strict scores (with bars)
- Hand × device IoU percentages
- Work / present streak timers vs thresholds
- Top 20 YOLO detections sorted by confidence
- Current processing FPS

The dashboard polls `/state` every 500 ms; the MJPEG `/stream` is push-driven, so it's as smooth as the pipeline can produce frames.

## 7. (Optional) Windows Firewall

If your Mac can't reach port 8000, allow it through Windows Firewall:
```powershell
New-NetFirewallRule -DisplayName "CV_Monitoring dashboard" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

## 8. (Optional) Run headless / on boot

To skip the OpenCV window entirely (recommended for a dedicated GPU box):
```powershell
python workbench_activity.py --web --no-show
```

To auto-start on boot, use Task Scheduler with "At log on" trigger and action `powershell -WindowStyle Hidden -Command "cd C:\path\to\CV_Monitoring; .\.venv\Scripts\Activate.ps1; python workbench_activity.py --web --no-show"`.

## 9. Performance expectations

| Stream | YOLO models | RTX 5000 expected FPS |
|---|---|---|
| Sub (896×512) | yolo11s + yolo11s-pose + yolo11m | **40–80 FPS** |
| Main (~1080p) | yolo11s + yolo11s-pose + yolo11m | **20–35 FPS** |

If the dashboard still feels laggy after this setup, the bottleneck is more likely the RTSP stream itself or the network than the pipeline. Try:
- Lower `YOLO_IMGSZ` from 1280 to 960
- Lower `YOLO_DEVICES_IMGSZ` from 960 to 640
- Set `FRIGATE_HTTP_FPS=10` (it controls the JPEG encode interval in the dashboard buffer; lower = less network usage)

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | You installed the CPU wheel — reinstall with `--index-url https://download.pytorch.org/whl/cu124` |
| `nvidia-smi` not found | NVIDIA driver isn't installed — get the Studio Driver from nvidia.com |
| RTSP "Connection refused" | The camera password or path is wrong — test with `ffplay "rtsp://..."` |
| Dashboard loads but no video | Wait 5–10 s for the first inference + JPEG encode; check the PowerShell terminal for errors |
| Web page reachable on Windows itself but not from Mac | Windows Firewall — run the `New-NetFirewallRule` command in step 7 |
| Out of GPU memory | Lower `YOLO_DEVICES_IMGSZ=640` in `.env`, or switch `YOLO_MODEL=yolo11n.pt` (nano) |
