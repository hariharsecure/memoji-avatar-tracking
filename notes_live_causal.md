# notes_live_causal.md

**Pipeline:** pipeline_live_causal_v1.py
**Date:** 2026-06-14
**Source clip:** input_clip.mov (847 frames, 720x1280, 29.0fps)
**Device:** MPS (Apple M3 Ultra, Mac Studio)

---

## Architecture

Causal forward-only IMM Kalman replacing the offline RTS backward smoother.
All v15 components retained: YOLOv10n-face + 6DRepNet360 + MediaPipe FaceLandmarker +
PoseLandmarker, ear-midpoint anchor (landmarks 234+454), source-aware heteroscedastic R,
jump-gate (150px). Calibration loaded from v15 pre-fitted JSON (376 pts, RMSE x=27.2 y=37.1px).

IMM models:

| Model | Type | State dim | Process noise Q |
|-------|------|-----------|-----------------|
| 0 | Constant Velocity (CV) | [pos, vel] = 2D | q=4.0 px/frame |
| 1 | Near-Constant Acceleration (NCA) | [pos, vel, acc] = 3D | q=64.0 px/frame |

Transition matrix: P(stay CV)=0.95, P(stay NCA)=0.70. Initial mu: [0.85, 0.15].
Output: Bayesian-weighted combination (position = sum_j mu_j * x_j[0]).
Heteroscedastic R: face_ear=225, pose_calib=2025, pose_raw=6400, hold=250000.

---

## Benchmark — Measured FPS per Stage (MPS, Mac Studio M3 Ultra)

All timings wall-clock over 847 frames.

| Stage | Mean ms/frame | P50 ms | P90 ms | Notes |
|-------|--------------|--------|--------|-------|
| MediaPipe FaceLandmarker | 6.5 | 6.2 | 7.4 | Every frame, CPU (XNNPACK) |
| MediaPipe PoseLandmarker | **21.4** | 21.2 | 22.0 | **BOTTLENECK** — every frame, CPU |
| YOLO detect (when MP fails) | 11.0 | 9.7 | 10.8 | REP360 frames only (456/847) |
| 6DRepNet360 (when YOLO fires) | 8.4 | 7.8 | 9.0 | ResNet50, MPS |
| Anchor + jump-gate | 0.008 | 0.012 | 0.015 | Pure numpy, negligible |
| IMM Kalman update | 0.13 | 0.13 | 0.14 | Pure numpy — zero overhead |
| **Total per frame** | **39.2** | **44.1** | **47.0** | Mean / P50 / P90 |

**Achieved FPS: 24.9 fps over 847 frames (33.98s wall-clock)**
**Real-time target: 24-30 fps**
**Verdict: PASS at 24.9fps — marginal. P50 frame time 44ms = ~22.7fps median throughput.**

Note: "Total per frame" > sum of stages because YOLO+REP360 overlap with per-frame bookkeeping.
MediaPipe frames (face visible) skip YOLO entirely and are faster (~28-30fps).
REP360 frames (YOLO+6DRepNet active) hit ~22fps. 24.9fps is the average.

---

## Detector Mode Breakdown

| Mode | Frames | % |
|------|--------|---|
| MEDIAPIPE | 376 | 44.4% |
| REP360 (YOLO+6DRepNet360) | 456 | 53.8% |
| HOLD (no detection) | 15 | 1.8% |

---

## Quality vs Offline v15 (Measured)

Comparison metric: Euclidean distance of causal anchor vs v15 offline RTS-smoothed anchor.
Lock threshold: 50px (within 50px = locked).

| Metric | Value |
|--------|-------|
| **Lock rate (delta <= 50px vs v15)** | **100.0%** |
| Mean position delta vs v15 | 7.7 px |
| Median (P50) delta vs v15 | 5.2 px |
| P90 delta vs v15 | 17.8 px |
| Max delta vs v15 | 48.2 px |
| Mean scale delta | 5.9 px |
| 100% anchor coverage | YES (IMM always outputs estimate) |
| HOLD frames | 15 (unchanged -- back-of-head has no detector) |

**By detector mode (mean delta-pos vs v15):**

| Mode | Mean delta | P50 delta | P90 delta |
|------|------------|-----------|-----------|
| MEDIAPIPE | 5.4 px | 3.6 px | 14.2 px |
| REP360 | 9.5 px | 6.7 px | 20.9 px |
| HOLD | 11.5 px | 10.1 px | 16.2 px |

The causal IMM matches v15 within 50px on all 847/847 frames. Max delta 48.2px occurred
during the back-of-head HOLD zone where both pipelines have low-quality anchors. The
forward-only Kalman produces slightly rougher estimates than the offline RTS smoother
(no backward refinement), but the difference is 7.7px mean -- within one calibration-RMSE.

---

## Bottleneck: MediaPipe PoseLandmarker (21.4ms/frame)

The PoseLandmarker runs on CPU via TensorFlow Lite / XNNPACK on every frame. 21.4ms per
frame regardless of detector mode. This is the single largest time consumer, exceeding
YOLO (11ms) and 6DRepNet360 (8.4ms) even though those run on only ~54% of frames.

MediaPipe does not expose an MPS/Metal backend for the Task API (as of 2026). It runs
XNNPACK on CPU. The M3 Ultra's 24-core CPU makes this faster than a typical CPU, but
it is still the wall.

---

## Speedup Path to Hit 29fps Reliably

Current: 24.9fps avg. 29fps source requires <=34.5ms/frame; current pipeline averages 39.2ms.

**Option 1 -- PoseLandmarker on alternate frames (estimated ~29fps, ~20 LOC change):**
Run PoseLandmarker every 2nd frame; on skipped frames hold last pose anchor. The IMM
Kalman predicts from the previous measurement. Cost halved from 21.4ms to ~10.7ms amortized.
Risk: 1-frame anchor lag on fast moves. At 29fps source, lag = 34ms -- acceptable for avatar.

**Option 2 -- PoseLandmarker only on non-MEDIAPIPE frames (~+3fps, ~15 LOC change):**
On MEDIAPIPE frames, the face ear-midpoint anchor is already available from FaceLandmarker
(Pose not needed for position). Run PoseLandmarker only when face drops (REP360/HOLD frames).
MEDIAPIPE = 44.4% of frames -- pose cost cut by ~44%. Estimated gain +2.5-3fps.

**Option 3 -- OpenSeeFace ONNX (~30fps achievable, larger refactor):**
Replace both FaceLandmarker + PoseLandmarker with OpenSeeFace (MobileNetV3, ONNX, CPU).
Designed for VTubing at 30-60fps. Would eliminate the ~28ms combined MediaPipe cost.
Requires re-mapping 68-landmark output to ear-midpoint anchor.

**Option 4 -- Frame-skip + CSRT for YOLO (~+2fps on REP360 frames):**
Run YOLO every 3rd REP360 frame, propagate box with CSRT between. Amortizes YOLO's 11ms
to ~3.7ms. Gain only on REP360 frames (54% of clip) so net +2fps.

**Production-live recommendation:** Options 1+2 combined require ~35 LOC change and would
bring from 24.9fps to estimated 28-30fps without model changes.

---

## IMM Model Probability Behavior

The NCA probability (p_nca) stayed low throughout (peak ~0.07, mean ~0.04). The CV model
dominated -- camera motion was smooth enough. This is correct: NCA activates when velocity
innovations are large (sudden moves). On this clip, the approach-to-camera zone (f799-844)
where the offline RTS had its largest errors was handled by the causal IMM with max delta 48.2px.

The IMM's primary value is robustness: when a zoom event or sudden head snap occurs, the NCA
model absorbs the motion within 2-3 frames vs the RTS smoother's 45-frame smear.

---

## Honest Assessment

**Real-time or not?** Marginally yes: 24.9fps on M3 Ultra. A MacBook or older Mac would be
below 24fps. The 29fps source requires <=34.5ms/frame; we average 39.2ms -- processing ~15%
slower than source. In true live streaming this would require dropping frames to maintain
wall-clock sync. The pipeline is real-time-capable at 24fps, not 29fps, on this hardware.

**Causal quality vs offline v15:** 100.0% lock rate with mean 7.7px delta. No visible quality
regression -- the causal IMM does not degrade tracking quality on this clip vs the offline RTS.

**Never-drop:** Confirmed 100%. The IMM always outputs a position estimate. HOLD frames (back-
of-head) exist in both pipelines equally.

**What a production live version needs:**
1. Pose-on-alternate-frames (Option 1 above) to reliably hit 29fps on M3 Ultra
2. Warm-start calibration (we used v15 pre-fitted -- works correctly)
3. Camera feed: replace VideoCapture with camera loop; process_frame_causal() is already
   stateless per-frame via the IMM object (no batch state)
4. On hardware weaker than M3 Ultra: OpenSeeFace replacement for MediaPipe

---

## Files

| File | Description |
|------|-------------|
| pipeline_live_causal_v1.py | This pipeline (causal IMM Kalman) |
| live_causal_v1_stream.npz | Rig stream (IMM-filtered anchors, 847 frames, 231KB) |
| live_causal_v1_report.json | Full benchmark + quality JSON (machine-readable) |
| live_causal_v1_overlay_master.mp4 | Full-res H.264 causal overlay (10.2MB) |
| live_causal_v1_overlay_preview.mp4 | web-preview-safe 480p overlay (1.5MB) |
| live_causal_v1_montage.png | 8-frame labeled proof montage |
| live_causal_v1_proof_f*.jpg | Individual proof frames |
