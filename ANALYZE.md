# CV_Monitoring — System Analysis

A line-by-line guide to what the system does and what every number on the OpenCV window means.

---

## 1. What the system does (in one sentence)

It polls a single camera snapshot from **Frigate** every ~0.5 s, runs **YOLO11** to find people and four work devices (phone, laptop, keyboard, mouse), runs a **pose model** to locate each person's wrists/hands, computes whether the hands overlap a laptop or keyboard, smooths the result with time-based stability rules, and reports the camera's activity state as **IDLE → PRESENT → WORKING** with a "PHONE" override when a phone is detected in the hand.

---

## 2. Per-frame pipeline

Each polled frame goes through this pipeline (orchestrated in [workbench_activity.py](workbench_activity.py)):

| # | Stage | Module | What happens |
|---|---|---|---|
| 1 | **Fetch** | [frigate_http.py](frigate_http.py) | `GET {FRIGATE_BASE_URL}/api/{camera}/latest.jpg` → JPEG → BGR frame |
| 2 | **Pass 1 — People** | YOLO11s @ imgsz 1280 | Detects only class `0` (person) at low confidence `YOLO_PERSON_CONF=0.02` |
| 3 | **Pass 2 — Devices** | YOLO11m @ imgsz 960 | One inference for classes `[67 phone, 63 laptop, 66 keyboard, 64 mouse]` |
| 4 | **Pose** | YOLO11s-pose (default) or MediaPipe | 17 COCO keypoints per person; the largest/most-central person is **primary** |
| 5 | **Wrist/hand boxes** | `build_wrist_boxes()` | A square box per wrist, centered **past the wrist toward the fingers**, sized by the forearm length (elbow→wrist distance × `HAND_BOX_FOREARM_SCALE`) |
| 6 | **Retag devices** | `retag_*_detections()` | For each device, compute three overlap metrics against each hand box: **IoU%**, **hand coverage%**, **device coverage%**. If any threshold is met, the role becomes `phone` / `laptop_hands` / `keyboard_hands` / `mouse_hands` |
| 7 | **Stabilize** | `ObjectDetectionStabilizer` | A device must be seen for **N seconds at ≥X confidence** before it counts (e.g. `LAPTOP_STABLE_SECONDS=3`, `LAPTOP_STABLE_MIN_CONF=0.5`) |
| 8 | **Optional zoom** | `detect_phones_on_hand_regions()` | If `USE_PHONE_HAND_CROP=1`, re-runs YOLO on a tight crop around the primary hand boxes to catch small phones |
| 9 | **Score** | `FrameScore.compute_totals()` | Build the **MEDIUM** and **STRICT** scores |
| 10 | **State machine** | `ActivityTracker.update()` | Apply hysteresis timers → emit `IDLE` / `PRESENT` / `WORKING` |
| 11 | **Render** | [workbench_viz.py](workbench_viz.py) | Composite top banner + boxes + skeletons + score panel at the bottom |

---

## 3. The screen layout

The OpenCV window is **video frame on top + a black score panel stacked underneath**.

### 3.1 Top banner (over the video)

A black strip at the top with the current state, broken down as:

```
┌────────────────────────────────────────────────────────────┐
│ TITLE                                                       │  ← big colored text
│ subtitle (reason for the state)                             │
│ max hand-phone IoU: 27%      ← only when > 0                │
│ max hand-laptop IoU: 41%                                    │
│ max hand-keyboard IoU: 33%                                  │
└────────────────────────────────────────────────────────────┘
```

| Title text | Color | Meaning |
|---|---|---|
| `IDLE` | grey | No person, or scores below the *present* threshold |
| `PRESENT` | yellow | Person standing/visible in the ROI for ≥ `PRESENT_SECONDS`, but not actively working |
| `WORKING` | green | Hands on stable laptop/keyboard for ≥ `WORK_SECONDS` |
| `WORKING (held)` | orange-cyan | Was working, hands just left — held during `WORK_OFF_SECONDS` cooldown |
| `PHONE` | red | Stable phone overlaps a hand box (overrides WORKING) |

The two main scores `M=` and `S=` are shown only when the state is not `WORKING` / `PHONE` (when neither has triggered).

The three `max hand-X IoU` lines only appear when that overlap is > 0; they're the **best** percentage across all hand×device pairs this frame.

### 3.2 Boxes drawn on the video

Every YOLO detection is a colored rectangle with a label above it `<class> <conf> | IoU 28% (hand 45% phone 33%) overlaps hand in ROI`.

| Box color | Box thickness | Meaning |
|---|---|---|
| Blue, thick (4 px) | — | **Primary person** — the one used for scoring |
| Blue, normal (2 px) | — | Other person |
| **Green (thick)** | — | A device tagged `*_hands` (laptop_hands / keyboard_hands / mouse_hands) — drives `WORKING` |
| Cyan | 2 px | Laptop on the desk (stable, but not at hands) |
| Cyan/Purple | 2 px | Keyboard on the desk |
| Yellow | 2 px | Mouse on the desk |
| **Red (thick)** | — | Phone whose box overlaps a hand region (drives `PHONE` banner) |
| Black | 2 px | Phone visible but NOT in hand (desk phone) |
| Grey, thin (1 px) | — | `pending` device — seen but stability timer not yet met. Label shows `pending 1.4s` countdown |
| Dark green | 1 px | "Weak" tool — COCO class with low confidence but kept for context |
| Gold/cyan small squares | 2 px | The hand region boxes (`left hand`, `right hand`, `P2_left`, …) |
| Magenta dots + lines | — | Skeleton (from YOLO11-pose or MediaPipe) |
| Lime ROI rectangle | 2 px | Workbench region of interest (only drawn when ROI ≠ full frame) |

### 3.3 IoU lines between hand and device

When a hand-box ↔ device overlap is computed it's drawn as a **thin colored line** with text mid-line:

| Line color | Meaning |
|---|---|
| Red | hand ↔ phone |
| Orange | hand ↔ laptop |
| Pink | hand ↔ keyboard |
| Yellow | hand ↔ mouse |

Mid-line text reads `cell phone IoU 27%` etc.

### 3.4 Status text at the bottom of the video

A single line just above the score panel:

```
YOLO: 2 person(s) — primary = thick blue (WORKING score)
```

Or in red if no person is found:

```
YOLO: no person detected (try yolo11s.pt / lower YOLO_CONF)
```

### 3.5 Bottom score panel (black, ~230 px tall)

Seven labeled horizontal bars. Each bar fills proportionally to its 0.0–1.0 value, with a thin white tick at the threshold where applicable.

| Bar label | Value range | Source | What it measures |
|---|---|---|---|
| **person in ROI** | 0.00–1.00 | `FrameScore.person_in_roi` | Fraction of the primary person's bounding box that falls inside the workbench ROI. Contributes **0.35 × value** to MEDIUM. |
| **hands in ROI** | 0.00–1.00 | `FrameScore.hands_in_roi` (capped at 0.15 if phone-in-hand, floored at 0.45 if at computer work) | Fraction of the wrist points inside the ROI. Contributes **0.35 × value** to MEDIUM. Label changes to `hands in ROI (computer work)` or `hands in ROI (phone — not working)` |
| **active pose** *or* **activity (YOLO)** | 0.00–1.00 | `FrameScore.active_pose` | Pose heuristic — non-zero when the skeleton looks like a person doing something. Falls back to `0.45 + 0.35 × hands_in_roi` when only YOLO is available. Contributes **0.30 × value** to MEDIUM. |
| **MEDIUM (phase 2)** | 0.00–1.00 | `FrameScore.medium_score` = sum of the three weighted bars above | White tick = `SCORE_MEDIUM` (default **0.50**). Above this for `PRESENT_SECONDS` → person counts as PRESENT. |
| **named COCO tools (k/n)** | 0.00–1.00 | `FrameScore.named_tool_score` | k = tools detected in this frame, n = tools checked. Higher = more recognized tools near the hands/bench. Contributes **0.20 × value** to STRICT if any tool seen. |
| **bench clutter (N objs, not chairs)** | 0.00–1.00 | `FrameScore.bench_clutter_score` | Fallback — counts "stuff on the bench" (excluding chairs/tables) when zero named tools were seen. |
| **STRICT (phase 3)** | 0.00–1.00 | `FrameScore.strict_score` = MEDIUM + tool/clutter bonus | White tick = `SCORE_STRICT` (default **0.62**). |

Below the bars, one summary line:

```
persons: 1 | tools (green): 3 | weak: 1 | other: 2 | phone: 0
```

| Token | Meaning |
|---|---|
| `persons: N` | YOLO person count in frame |
| `tools (green): N` | Detections with role `tool_ok` |
| `weak: N` | Detections with role `tool_miss` (low conf but still drawn dimly) |
| `other: N` | Detections with role `bench_obj` (counted as clutter) |
| `phone: N` | Detections with role `phone` (in-hand only) |

Then a constant info line listing the COCO tool vocabulary (`backpack, book, bottle, ...`).

### 3.6 Timer line (bottom of the score panel)

```
timers: laptop/keyboard 4.2s (need 10.0s) | present 6.1s
```

| Field | Source | What it means |
|---|---|---|
| `laptop/keyboard X.Xs` | `tracker.work_surface_streak` | Continuous seconds the hands have been on a stable laptop/keyboard (and no phone). |
| `(need Y.Ys)` | `WORK_SECONDS` from `.env` | Required streak for state to flip to **WORKING**. |
| `present X.Xs` | `tracker.present_streak` | Continuous seconds the MEDIUM score has stayed above `SCORE_PRESENT` (=0.28 default). Used to flip to **PRESENT**. |

### 3.7 Legend (top-right of the video)

A small grey-text legend, ~20 lines, restating every color and the skeleton source (`YOLO11-pose (1 shown)` or `MediaPipe (1 shown)`), plus the total person count.

---

## 4. The numbers shown next to each detection

A typical YOLO box label looks like:

```
laptop 0.78 | IoU 41% (hand 56% laptop 38%) in person box overlaps hand in ROI
```

| Token | Meaning |
|---|---|
| `laptop` | YOLO class name (COCO) |
| `0.78` | Model confidence (0–1) |
| `IoU 41%` | Intersection ÷ Union of (hand box, device box) × 100 |
| `hand 56%` | Intersection ÷ hand-box area × 100 (how much of the hand is covered) |
| `laptop 38%` | Intersection ÷ device-box area × 100 (how much of the laptop is covered) |
| `in person box` | The device's center lies inside a YOLO person box (likely held) |
| `overlaps hand` | A hand box overlaps this device (drives the `*_hands` role retag) |
| `in ROI` | The device's center is inside the workbench ROI |
| `[hand-zoom]` | (Phones only) — this box came from the hand-region zoom pass, not the full frame |
| `pending 1.4s` | (Replaces the role) — detection seen, stability timer still running; needs to reach `<X>_STABLE_SECONDS` |

The decision to mark a device as **at hands** uses three percent thresholds simultaneously (any one passes → tagged), all configurable per device in `.env`:

| Device | IoU% min | Hand-coverage% min | Device-coverage% min |
|---|---|---|---|
| Phone | `WRIST_PHONE_MIN_IOU_PCT=8` | `WRIST_PHONE_MIN_WRIST_COV_PCT=12` | `WRIST_PHONE_MIN_PHONE_COV_PCT=15` |
| Laptop | `WRIST_LAPTOP_MIN_IOU_PCT=10` | `WRIST_LAPTOP_MIN_HAND_COV_PCT=15` | `WRIST_LAPTOP_MIN_LAPTOP_COV_PCT=12` |
| Keyboard | `WRIST_KEYBOARD_MIN_IOU_PCT=10` | `WRIST_KEYBOARD_MIN_HAND_COV_PCT=15` | `WRIST_KEYBOARD_MIN_KEYBOARD_COV_PCT=12` |
| Mouse | `WRIST_MOUSE_MIN_IOU_PCT=10` | `WRIST_MOUSE_MIN_HAND_COV_PCT=15` | `WRIST_MOUSE_MIN_MOUSE_COV_PCT=12` |

---

## 5. The state machine

Implemented in [workbench_logic.py:336 `ActivityTracker.update()`](workbench_logic.py#L336).

```
                            ┌──────────────┐
        person disappears   │              │   medium_score ≥ 0.50
        ┌────────────────── │     IDLE     │ ──────────────┐
        │                   │   (grey)     │               │
        │                   └──────────────┘               │
        ▼                                                  ▼
┌──────────────┐                                    ┌──────────────┐
│   (gone)     │                                    │   PRESENT    │
└──────────────┘                                    │   (yellow)   │
        ▲                                           └──────────────┘
        │                                                  │
        │     work_surface_streak ≥ WORK_SECONDS           │
        │     AND hands on laptop/keyboard                 │
        │     AND no phone                                 │
        │                                                  ▼
        │                                          ┌──────────────┐
        │       working_cooldown reaches 0 ────────│   WORKING    │
        │       AND hands have left                │   (green)    │
        └──────────────────────────────────────────└──────────────┘

PHONE override: at any state, if a stable phone overlaps a hand box,
the banner shows "PHONE" (red) and WORKING is suppressed.
```

Default thresholds (in `.env`):

| Env var | Default | What it controls |
|---|---|---|
| `WORK_SECONDS` | 10 | Continuous "hands on laptop/keyboard" seconds before flipping to WORKING |
| `WORK_OFF_SECONDS` | 4 | Grace period before WORKING ends after hands leave |
| `PRESENT_SECONDS` | 3 | Seconds above the present threshold before flipping to PRESENT |
| `SCORE_MEDIUM` | 0.50 | MEDIUM threshold to count as actively present |
| `SCORE_STRICT` | 0.62 | STRICT threshold (used in summary output, not state gating) |
| `SCORE_PRESENT` | 0.28 | MEDIUM threshold to start the present streak |

---

## 6. Terminal output

While running you'll see (compact mode):

```
[16:42:11] idle → present (M=0.54 S=0.74 persons=1 via strict)
[16:42:18] present → working (M=0.62 S=0.82 persons=1 via strict)
[16:42:35] working → present (M=0.51 S=0.61 persons=1 via medium)
```

Each transition prints **previous → next**, the two scores, the YOLO person count, and which tier (`medium` or `strict`) crossed the threshold this frame.

With `LOG_VERBOSE=1` (default) every frame additionally prints:

```
========== frame #142 16:42:18 ==========
[person] people_pass: 1 person(s)
[person] people_pass #1: person conf=0.78 role=person stable=False ... PRIMARY ...
[det] devices_pass: 3 box(es)
[det] devices_pass #1 laptop conf=0.81 role=laptop_hands stable=True streak=4.1s ...
[hands] wrist points (norm): 2
[hands] left primary: box=(412,288)-(478,354) center=(0.45,0.40)
[overlap] laptop: 2 pair(s)
[overlap] laptop #1: hand=left obj=laptop conf=0.81 IoU=42.1% hand_cov=58.3% obj_cov=33.7% ...
[score] state=working persons=1 poses=1 pose_yolo_fallback=False
[score] person_in_roi=0.92 hands_in_roi=0.84 active_pose=0.71 person_visible=True
[score] on_phone=False phone_near_hand=False on_laptop=True on_keyboard=False on_mouse=False at_computer_work=True
[score] IoU% phone=0 laptop=42 keyboard=0 mouse=0
[score] M=0.66 S=0.82 tools=2/3 named_tool=0.67
[score] near_hands: phone streak 0.0s | dets streak 4.2s
```

`LOG_VERBOSE=0` collapses this to ~1 line per device-log interval (~0.5 s).

---

## 7. Tuning knobs (most useful)

| Goal | Variable to change |
|---|---|
| Make WORKING trigger faster | Lower `WORK_SECONDS` (default 10) |
| Catch laptops further from the camera | Lower `YOLO_LAPTOP_CONF` (default 0.12) or raise `YOLO_DEVICES_IMGSZ` (default 960) |
| Hands have to *really* be on the keyboard before WORKING | Raise `WRIST_KEYBOARD_MIN_IOU_PCT` (default 10) |
| Restrict to a specific bench region | Set `WORKBENCH_ROI=x1,y1,x2,y2` as normalized 0–1 coords |
| Disable phone detection | `USE_YOLO_PHONE_PASS=0` |
| Slower / less CPU | Lower `FRIGATE_HTTP_FPS` (default 2), turn off `SHOW_ALL_TOOLS`, switch `YOLO_MODEL` to `yolov8n.pt` |

---

## 8. TL;DR — reading the screen at a glance

1. **Look at the banner.** Color + title = state. (Grey IDLE, yellow PRESENT, green WORKING, red PHONE.)
2. **Look at the timers line** under the score panel: the `X.Xs / Y.Ys` count is the live progress toward WORKING.
3. **Look at the green/red boxes.** Green-thick = a device the system thinks is being used. Red-thick = phone in hand.
4. **Look at the MEDIUM and STRICT bars.** If they're filled past the white tick, the person is registered as actively present.
5. **The IoU percentages in the banner** are the strongest hand-device overlap this frame; they explain *why* a device just turned green or red.
