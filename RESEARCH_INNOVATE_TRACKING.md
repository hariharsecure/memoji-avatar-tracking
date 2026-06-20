# Tracking Architecture Research & Innovation Report

**Date:** 2026-06-14
**Stack under review:** YOLOv10n-face + 6DRepNet360 + MediaPipe Pose anchor + RTS Kalman smoother (v13)
**Status at review:** 94% locked (grid-confirmed), 6% residual failure in three zones

---

## 1. Architecture Critique — Where Is the Pipeline Fundamentally Limited?

### 1a. The Paradigm: Per-Frame Detect + Smooth

The current design is a **detect-every-frame + offline smooth** pipeline:

1. Forward pass: per frame, run MediaPipe FaceLandmarker → if fail, run YOLOv10n-face → feed crop to 6DRepNet360 → extract head pose.
2. Pose-anchor pass: per frame, run MediaPipe PoseLandmarker → extract ear/shoulder midpoint.
3. Calibration pass: fit yaw-conditioned linear offset between face anchor and pose anchor.
4. Smooth pass: RTS forward-backward Kalman with constant-velocity model.

This is a reasonable offline batch paradigm but it has five structural weaknesses.

**Weakness 1 — Detector-switch seam causes anchor discontinuities.**
The pipeline switches between three anchor regimes (MediaPipe face → pose_calib → pose_raw) with fundamentally different reference points:
- MediaPipe anchor = nose-tip projected via OpenGL intrinsics (FOCAL_LEN=700, uncalibrated)
- Pose anchor = ear midpoint in pixel space
- At extreme pitch (f437 chin-up) these can diverge 50-100px or more

The calibration tries to patch this with a yaw-conditioned linear offset (RMSE_x=35.7px, RMSE_y=67.4px — large relative to a 720px frame). The patch is lossy because:
- The residual comes from pitch not being in the calibration model
- Training was done on only 376 frontal frames where pitch was near 0
- The linear sin/cos(yaw) model cannot represent the nose-to-ear displacement as a function of both yaw AND pitch

**Weakness 2 — 6DRepNet360 degrades at true back-of-head (yaw ~±160°).**
6DRepNet360 is trained on 300W-LP + Panoptic. Both datasets underrepresent true back-of-head orientations. The model returns a plausible-but-wrong rotation at yaw ~180°. On this clip, the 15 HOLD frames are confirmed gaps where no face crop is visible — 6DRepNet360 simply has nothing to run on. The avatar is held at the last valid rotation, which for a back-of-head turn is ~170° yaw — visually acceptable but not accurate.

**Weakness 3 — The RTS smoother assumes Gaussian measurement noise.**
The pipeline feeds three qualitatively different noise sources (MP face projection, pose-calib, pose-raw) into a single Kalman track with one noise model (Q=4px). This violates the Kalman assumption: the "noise" at a mode transition is not Gaussian — it is a structured step-discontinuity from reference point change. The result is the f799-844 Zone 2 overshoot: the backward pass can't distinguish "person moved 100px" from "anchor reference changed 100px" and smears the discontinuity across 45 frames.

**Weakness 4 — Constant-velocity Kalman fails on scale discontinuities (zoom).**
The adaptive scale (9.0 * 100/head_scale_px, clamped [4,20]) uses head_scale_px from the pose anchor. When the camera zooms (f828 rapid zoom), head_scale_px jumps discontinuously. A constant-velocity Kalman has no scale model except "keep moving at current speed" — so it extrapolates past the discontinuity. v14's FIX 3 (segment-split at scale jumps >80px) is a correct patch but is reactive, not predictive.

**Weakness 5 — No temporal memory of face identity across detector switches.**
The pipeline treats each frame independently. When YOLO detects a face crop and 6DRepNet360 estimates a pose, it does not verify that the detection is the same person visible in adjacent frames (no IoU linking, no re-identification). This matters less on a single-person clip but means any detector confusion (another person entering frame, mirror, TV in background) would be silently accepted.

---

### 1b. Where Each Failure Mode Lives

| Failure zone | Root cause | Structural fix class |
|---|---|---|
| f437 chin-up anchor jump | Nose-tip vs ear-midpoint reference gap + no pitch term in calibration | Unified reference point; pitch-aware calibration; jump gate |
| f799-844 RTS overshoot/undershoot | Kalman fed mixed-reference measurements; backward pass propagates reference-change as trajectory event | Source-aware measurement noise; forward-fallback; segment split |
| Back-of-head 15 frames (HOLD) | No face visible, no detector can help; last-pose hold is the best available option | Head segmentation mask for position; avatar back-of-head texture for pose |
| Adaptive scale instability during profile turns | Ear-span vanishes at yaw~90°; shoulder proxy is coarse (±20px) | Depth from focal-length + face size history; head-size prior |
| Yaw calibration RMSE 35-67px | Missing pitch term; trained on too-narrow pose envelope; uncalibrated FOCAL_LEN | Geometric model (known head geometry); calibrated intrinsics |

---

### 1c. Is "Per-Frame Detect + Smooth" the Right Paradigm?

For an offline batch run on a short clip (847 frames), it works adequately. The paradigm's ceiling is set by how accurately each detector localizes the head center on each frame in isolation. Three alternative paradigms are worth considering:

**Paradigm A — Video tracker (motion-model propagation with detect-on-low-confidence)**
Detect once on a high-confidence frame, propagate with an appearance-based short-term tracker (CSRT, nanotrack, ByteTrack-head), only re-detect when tracker confidence drops. This eliminates the per-frame detector noise and the reference-switch problem: the tracker tracks one stable bounding box and the reference point never changes between modes. Downside: trackers drift over long sequences and need a good initial detection.

**Paradigm B — Temporal landmark smoothing (e.g., Kalman on 68/478 landmarks directly)**
Run a face landmark detector that produces landmarks for each frame, then smooth the 2D landmark positions with an individual Kalman filter per landmark. This eliminates the need for a "which anchor to use?" arbitration step — the landmark set itself encodes face position, scale, and pose. When the face goes out of range, the tracker predicts landmark positions from the Kalman velocities. At back-of-head, fall back to body pose. OpenSeeFace uses this approach (MobileNetV3 + Kalman) and is specifically designed for wide-angle tracking stability.

**Paradigm C — 3DMM-based tracker (FLAME/3D morphable model with per-frame energy minimization)**
Fit a 3D head model to observed landmarks and silhouettes on every frame. The 3D model naturally handles large-yaw and pitch changes because it is a 3D object. At back-of-head, the model can be rendered from behind. VHAP and video-head-tracker are PyTorch FLAME-based trackers. Downside: these are research-grade, not real-time, and VHAP requires CUDA 12.1 compilation — not MPS-runnable as-is.

**Verdict on paradigm:** Per-frame detect + smooth is correct for offline batch and for MPS-only hardware. The key innovation is not changing the paradigm but making the detect step produce a **single stable reference point** regardless of which detector fires — eliminating the reference-switch problem. This is the highest-leverage architectural fix.

---

## 2. Research Findings — SOTA Models Runnable on MPS

### 2a. Head Detectors

**YOLOv8n trained on SCUT-HEAD (Abcfsa/YOLOv8_head_detector)**
- SCUT-HEAD is a 4405-image, 111K-head dataset with occlusion annotations covering back/side heads
- Pre-trained nano.pt and medium.pt weights available from GitHub
- Runs via Ultralytics (already installed), runs on MPS natively
- Key difference from yolov10n-face: trained on HEAD boxes (including back of head and occluded heads) not FACE boxes
- Expected to detect at higher yaw angles and fire on back-of-head frames where yolov10n-face returns nothing
- Source: https://github.com/Abcfsa/YOLOv8_head_detector, https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release

**YOLOv8n trained on CrowdHuman (head class)**
- CrowdHuman dataset has "head" and "full body" bounding boxes including heavily occluded heads in dense crowds
- Back-of-head is included in the annotation scheme
- Source: multiple community fine-tunes on Roboflow/HuggingFace; the AbelKidaneHaile/Reports repo uses YOLOv8 on this

**img2pose (CVPR 2021, vitoralbiero/img2pose)**
- Faster R-CNN based: detects face AND estimates 6DoF pose simultaneously from the whole image, without a separate face crop step
- Removes the two-stage YOLO → 6DRepNet pipeline with a single-stage equivalent
- Trained on WIDER FACE; handles wider-than-frontal yaw
- Runs on MPS: standard PyTorch ResNet50 backbone, no CUDA-specific ops reported
- Download: HuggingFace or GitHub, model weights ~90MB
- Limitation: still face-based (not head-based), degrades at true back-of-head similar to 6DRepNet360
- Source: https://github.com/vitoralbiero/img2pose, https://arxiv.org/pdf/2012.07791

### 2b. Head Pose Estimators

**6DRepNet360 (current, thohemp/6DRepNet360)**
- Already deployed. IEEE TIP 2024. MIT license. Full 360° trained.
- Best available open-weight pose estimator on MPS for face-visible frames.
- Limitation: degrades gracefully at yaw~160° but returns arbitrary values at true back-of-head.

**Latent Space Regression for occlusion-robust pose (LSR, arxiv 2403.20251)**
- Designed specifically for occluded head pose estimation. ResNet-50 backbone + latent space regression.
- Multi-loss training makes it more robust when part of the face is hidden.
- Research paper only — no confirmed pre-trained weight release yet (as of this search).
- Source: https://arxiv.org/pdf/2403.20251

**Bidirectional Regression for 6DoF Head Pose (arxiv 2407.14136)**
- Temporal bidirectional regression: incorporates adjacent frame pose to stabilize single-frame estimate
- Directly targets the temporal smoothness problem: the model itself produces temporally consistent pose without a separate smoother
- Research paper (2024); check for weight release
- Source: https://arxiv.org/pdf/2407.14136

### 2c. Temporal / Video Trackers

**OpenSeeFace (emilianavt/OpenSeeFace)**
- MobileNetV3-based, ONNX format, runs 30-60fps on CPU, no GPU required (MPS would be faster)
- Four model variants: quality/speed trade-off
- Specifically designed for VTubing (avatar driving) — directly competes with this stack's end goal
- Produces 68 face landmarks + optional head pose
- Per the README: "keeps tracking faces through a very wide range of head poses with relatively high stability"
- Key advantage over current stack: runs as a video tracker (frame-to-frame tracking state), not per-frame detection; more stable at occlusion boundaries
- Does NOT handle true back-of-head (no face visible)
- License: Apache 2.0
- Source: https://github.com/emilianavt/OpenSeeFace, https://deepwiki.com/emilianavt/OpenSeeFace

**Optical flow + landmark bridge**
- Standard approach: when face detector confidence drops below threshold, switch to Lucas-Kanade sparse optical flow tracking of the last-known facial landmarks. Provides sub-pixel stable positions during 5-15 frame gaps. Fails for long occlusions.
- OpenCV calcOpticalFlowPyrLK is CPU-only but fast; MPS-based flow (via kornia or torchvision) available.

**ByteTrack / SORT applied to head bounding boxes**
- Rather than tracking face landmarks, track the head bounding box as an object track.
- ByteTrack assigns detection boxes across frames using IoU + Kalman. Works at profile because head box is larger than face box and present at higher yaw.
- Does not require any additional model download; ByteTrack is in ultralytics already.
- Limitation: box center is not a stable sub-pixel anchor; still needs 6DRepNet for pose.

### 2d. 3DMM / FLAME-Based Trackers

**video-head-tracker (philgras/video-head-tracker)**
- Python library, FLAME 3DMM, PyTorch
- Tracks the 3D head shape over video frames — head position + shape + pose
- Fully MPS-runnable: pure PyTorch, differential renderer
- Benefit: the 3D model anchor is geometrically consistent — the "center" of the FLAME head is always the same skull point regardless of face vs back view
- Limitation: offline (not real-time), requires initialization with visible face, computationally heavy (seconds per frame on CPU)
- Source: https://github.com/philgras/video-head-tracker

**VHAP (ShenhanQian/VHAP)**
- More complete pipeline: FLAME 2020, video tracking with appearance priors
- REQUIRES CUDA 12.1 for compilation of custom rasterizer — NOT MPS-runnable without significant porting
- Source: https://github.com/ShenhanQian/VHAP

---

## 3. Prioritized Innovation Proposals

Ordered by impact-to-effort ratio. Each proposal targets a specific confirmed failure mode.

---

### PROPOSAL 1 — Head-Class Detector to Replace Face-Class Detector for Anchor

**Failure mode fixed:** Back-of-head HOLD frames (15 frames), profile boundary overshoot, Zone 1 f435-439 anchor jump
**Core idea:** Replace yolov10n-face (trained on face visibility) with yolov8n-head (trained on SCUT-HEAD / CrowdHuman) as the fallback detector. A head detector fires on head shape — which is visible even from behind. At yaw~180° a face detector sees nothing; a head detector sees the round occlusion of the back of the skull.
**What this buys:**
- The 15 HOLD frames could become REP360 frames if the head box gives enough crop for pose estimation
- At yaw~160°, pose from the back is ill-defined but the position anchor (head box center) would be correct — solving the "where is the head" problem independently of "which way is the head"
- Decouples position anchor (head box) from pose estimator (6DRepNet360 on face crop)
**MPS-runnable:** YES — Ultralytics YOLOv8 already on MPS
**Effort:** Low — download yolov8n-head weights (6MB), drop in as a second fallback tier below yolov10n-face. Architecture: MP-face → YOLO-face → YOLO-head → pose-anchor
**Download needed:** Abcfsa/YOLOv8_head_detector (nano.pt, ~6MB) from GitHub. Or fine-tune yolov8n on SCUT-HEAD in ~2h on Studio GPU.
**Honest assessment:** Worth building. This directly addresses HOLD frames, which are the one place the pipeline has no detection at all. Even partial coverage (detecting head center from behind) turns a zero-information frame into a low-confidence-position frame, which the Kalman can use productively.

---

### PROPOSAL 2 — Source-Aware Measurement Noise in the Kalman (Heteroscedastic Smoother)

**Failure mode fixed:** Zone 2 f799-844 RTS backward overshoot (mean 98.9px error); mode-transition seams
**Core idea:** The RTS smoother currently uses a single observation noise R for all frames regardless of anchor source. In reality:
- MediaPipe face anchor: position error ~10-20px (MP FaceLandmarker is accurate when it fires)
- pose_calib anchor: position error ~30-50px (calibration RMSE = 35-67px)
- pose_raw anchor: position error ~50-80px (raw pose, no calibration)
- HOLD (no detection): infinite measurement noise (no observation)

The fix is to assign per-frame R values (observation noise covariance) to the Kalman based on anchor_source and anchor_confidence:
```
R_mp_face   = diag([15², 15²])  # MediaPipe face, high-trust
R_pose_cal  = diag([45², 70²])  # pose_calib, RMSE-derived
R_pose_raw  = diag([70², 90²])  # pose_raw, low-trust
R_hold      = diag([500², 500²]) # no detection — near-infinite, let Kalman predict freely
```
This makes the smoother weight each measurement by its true reliability. At a mode-switch from MP-face to pose_calib, the Kalman correctly assigns higher uncertainty to the post-switch measurements and does not violently pull adjacent high-confidence frames toward the noisy measurement.
**MPS-runnable:** YES — pure numpy/scipy, no model download
**Effort:** Low to Medium — the Kalman loop exists; add R_t as a per-frame array and pass it to the filter update step. ~50 lines of code change in the smoother.
**Download needed:** None
**Honest assessment:** This is the highest-value code fix available right now. The Zone 2 overshoot (46 frames, mean 98.9px) is almost entirely caused by the smoother treating a pose_calib measurement identically to a MP-face measurement. This fix requires no new model, no new dependency, and addresses the worst confirmed failure zone. Implement in v15.

---

### PROPOSAL 3 — Geometric Anchor Unification (Replace Nose-Projection with Midpoint-of-Ear-Span as Universal Reference)

**Failure mode fixed:** f437 chin-up anchor jump (56px outlier); systematic ~40px frontal offset visible in grid confirm (mediapipe_face mean offset 56.5px vs pose_calib 26.9px)
**Core idea:** The grid confirm reveals that the MP-face anchor (nose-tip projected via uncalibrated FOCAL_LEN=700) produces a LARGER median offset from the YOLO face bbox center (56.5px) than the pose_calib anchor (26.9px). This is backwards — the face anchor should be the ground truth, not the noisy one. The problem is the reference point:
- YOLO bbox center = center of visible face area
- MP nose-tip projection = nose-tip reprojected through an assumed focal length
- Ear midpoint = anatomically stable, does not shift dramatically with pitch

Fix: for MP-face frames, ALSO compute the ear midpoint from MediaPipe face landmarks (ears are landmarks 234, 454 in the 478-point mesh) and use EAR MIDPOINT as the anchor, not the nose-tip projection. This makes MP-face and pose anchors agree on the same anatomical reference point. The yaw-conditioned calibration can then be retrained with a much smaller residual (the main residual was from the reference-point mismatch, not from actual head position error).
**MPS-runnable:** YES — MediaPipe already runs, just extract different landmarks
**Effort:** Low — in face_head_anchor_from_mp(), replace the nose-tip projection with the normalized mean of face landmarks 234 and 454 (ears), projected to pixel coords via the same MP world space. ~20 lines of change.
**Download needed:** None
**Honest assessment:** Worth building. The face anchor RMSE (35-67px) is partly reference-point noise, not tracking error. Unifying the reference point would shrink the calibration RMSE and reduce the need for a large jump-gate threshold. This is a clean geometric fix that costs almost nothing.

---

### PROPOSAL 4 — IMM Kalman (Interacting Multiple Model) for Scale and Zoom Discontinuities

**Failure mode fixed:** f799-844 Zone 2 rapid-zoom scale discontinuity; v14 FIX 3 segment-split is reactive (manual threshold)
**Core idea:** Instead of splitting the smoother at detected discontinuities, run an IMM Kalman that maintains two parallel models:
- Model 1: constant velocity (CV) — good for smooth head motion
- Model 2: constant acceleration (CA) or near-white-noise / "maneuvering" model — good for rapid camera zoom or subject jump

The IMM framework continuously weights the two models by their likelihood given the observations. During smooth motion, Model 1 dominates (low process noise, accurate prediction). When a zoom event occurs, observations diverge from the CV prediction, the IMM upweights the CA/maneuver model (high process noise, lets measurements dominate), and the smoother adapts in 2-3 frames rather than smearing the discontinuity over 45 frames.

IMM for position tracking is well-established in aerospace tracking and has a standard 4-step algorithm (interaction, filtering, weight, combination). A 2D (cx, cy) IMM with two models is ~150 lines of numpy.

**MPS-runnable:** YES — pure numpy, no model download
**Effort:** Medium — more involved than Proposal 2 (need to implement IMM loop, tune model noise parameters, validate on the zoom zone). But the core math is deterministic and well-tested in other domains.
**Download needed:** None
**Honest assessment:** Worthy for v16 or later. v14 FIX 3's segment-split solves the immediate problem. IMM would be the principled upgrade: it handles zoom without needing to hard-code a "scale jump threshold" and generalizes to other motion discontinuities (subject standing up, camera pan). Research confirms IMM with CV+CA models handles "start-stop" motion well (arxiv 2502.09672). Build after the simpler fixes are validated.

---

### PROPOSAL 5 — OpenSeeFace as a Parallel Video-Tracker Lane for Occlusion Boundary Bridging

**Failure mode fixed:** All occlusion-boundary transitions (gap entry/exit jitter); blendshape instability during profile turns
**Core idea:** Run OpenSeeFace (MobileNetV3, ONNX, CPU/MPS) in parallel with the current MediaPipe pass. OpenSeeFace operates as a video tracker (frame-to-frame state), not a per-frame detector. At gap boundaries (f417, f484, f521, f575, etc.) where MediaPipe drops out and YOLO/6DRepNet360 takes over, OpenSeeFace's tracker will often continue propagating a stable landmark estimate for 5-15 more frames before it also loses the face.

Use OpenSeeFace as a third anchor source with confidence tied to its internal tracker confidence:
- OpenSeeFace high-confidence + MP-face both present: average (or weight by confidence)
- OpenSeeFace tracking + MP dropped: use OpenSeeFace landmarks as primary until confidence < threshold
- OpenSeeFace dropped too: current pose_calib fallback

This effectively adds a 5-15 frame "soft landing" at gap boundaries — exactly where the current pipeline shows its worst seam artifacts.

**MPS-runnable:** YES — ONNX runtime, CPU optimized, can run on MPS via onnxruntime-silicon
**Effort:** Medium-High — requires integrating a separate tracking process; OpenSeeFace outputs 68 landmarks not 478, so blendshape estimation would need to bridge the two landmark sets. However, for the ANCHOR purpose (where is the head center) the 68-landmark output is sufficient. The blendshapes can remain MediaPipe-derived.
**Download needed:** OpenSeeFace models (~50MB total, 4 ONNX variants); `pip install onnxruntime` (already likely installed)
**Honest assessment:** High value but higher integration cost. Build after Proposals 1-3 are shipped and the residual from boundary seams is measured. The VTubing community has validated OpenSeeFace as the reference implementation for this exact problem (avatar overlay from webcam through wide head poses). Worth a 2-day integration spike.

---

## 4. What Is NOT Worth Building (and Why)

**FLAME / VHAP 3DMM tracker:** Requires CUDA 12.1 compilation. Not MPS-runnable without porting the rasterizer (weeks of work, uncertain outcome). Set aside for a future CUDA machine.

**WHENet (full-range head pose, EfficientNet + YOLOv3):** More complex pipeline than 6DRepNet360, no pip install, less tested on MPS. 6DRepNet360 is already doing what WHENet targets.

**3DDFA_V2:** No pip install, requires build.sh + libomp on Mac (fragile), max ~90° yaw. Does not extend our coverage at back-of-head.

**Optical flow as primary tracker:** Useful for 5-15 frame bridge but drifts on longer gaps. Not a replacement for the YOLO/pose fallback; add as a supplementary lane only if gap-boundary seams remain after Proposals 1-3.

**Bidirectional temporal pose regression (arxiv 2407.14136):** Interesting — the model itself learns to produce temporally consistent pose without a smoother. But no confirmed weight release as of this search. Watch for release; if weights drop, this could replace 6DRepNet360 + RTS smoother in one step.

---

## 5. Recommended Build Sequence

| Priority | Proposal | Expected improvement | Effort |
|---|---|---|---|
| 1 (ship in v15) | Proposal 2: Source-aware heteroscedastic Kalman R | Zone 2: 98.9px mean → estimated <20px | Low — 50 LOC change |
| 2 (ship in v15) | Proposal 3: Ear-midpoint anchor unification | Frontal calibration RMSE: 35-67px → estimated 10-20px; eliminate f437 reference-jump root cause | Low — 20 LOC change in MP anchor extraction |
| 3 (ship in v15-v16) | Proposal 1: YOLOv8n-head as third detector tier | HOLD frames: 15 → estimated 5-8 (gain position anchor on back-of-head frames) | Low — download 6MB weights, add fallback tier |
| 4 (v16) | Proposal 4: IMM Kalman for zoom | Handles rapid-zoom/stand events without manual segment split | Medium — 150 LOC numpy IMM |
| 5 (v16+) | Proposal 5: OpenSeeFace parallel lane | Smooth gap-boundary seams, 5-15 frame soft landing | Medium-High — ONNX integration |

---

## 6. One-Line Architectural Thesis

The current pipeline's residual errors are almost entirely caused by two things: (1) the Kalman smoother is fed heterogeneous measurements from different reference points and assigns identical noise to all of them, and (2) the anchor reference point changes between detector modes. Fix the reference point (Proposal 3) and fix the noise model (Proposal 2), and the smoother will work correctly without needing special-case gates or segment splits.

---

## Sources

- OpenSeeFace: https://github.com/emilianavt/OpenSeeFace / https://deepwiki.com/emilianavt/OpenSeeFace
- 6DRepNet360 (IEEE TIP 2024): https://github.com/thohemp/6DRepNet360
- img2pose (CVPR 2021): https://github.com/vitoralbiero/img2pose / https://arxiv.org/pdf/2012.07791
- Latent Space Regression occlusion-robust HPE: https://arxiv.org/pdf/2403.20251
- Bidirectional 6DoF temporal regression: https://arxiv.org/pdf/2407.14136
- SCUT-HEAD dataset: https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
- YOLOv8 head detector (SCUT-HEAD trained): https://github.com/Abcfsa/YOLOv8_head_detector
- video-head-tracker (FLAME, PyTorch): https://github.com/philgras/video-head-tracker
- VHAP (FLAME 2020 pipeline): https://github.com/ShenhanQian/VHAP
- IMM-MOT Kalman framework: https://arxiv.org/pdf/2502.09672
- Deep learning head pose survey (Springer 2024): https://link.springer.com/article/10.1007/s10462-024-10936-7
- Head pose estimation GitHub topics: https://github.com/topics/head-pose-estimation
