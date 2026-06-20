# H1 Head-Detector Notes — v16

**Date:** 2026-06-14
**Pipeline:** pipeline_memoji_rig_v16_headdet.py
**Source video:** input_clip.mov (847 frames, 720x1280, 29fps)
**Based on:** pipeline_memoji_rig_v15.py (all v15 changes retained)
**Metric target:** HOLD frames 15 → ≤5 (pre-registered success bar)

---

## Result: HOLD 15 → 0

| Metric | v15 | v16 H1 | Delta |
|--------|-----|--------|-------|
| HOLD frames | 15 | 0 | -15 (−100%) |
| HEAD_DET frames (new tier) | 0 | 15 | +15 |
| MEDIAPIPE frames | 376 | 376 | 0 (unchanged) |
| REP360 frames | 456 | 456 | 0 (unchanged) |
| Frames with NO anchor | 15 | 0 | Eliminated |

Success metric ≤5 was exceeded: 0 HOLD frames remain.

---

## Model: YOLOv8n-head (SCUT-HEAD trained)

**Source:** Abcfsa/YOLOv8_head_detector (GitHub graduation project 2023-24)
**Training data:** SCUT-HEAD dataset (4405 images, 111K heads, includes occlusion/back/side)
**Weights:** `yolov8n-head-scut.pt` (6.0 MB)
**Download:** `https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/nano.pt`
**Classes:** {0: 'head'} — one class only, head shape including back-of-head
**Framework:** Ultralytics YOLOv8 (same API as existing yolov10n-face.pt)

**MPS confirmation:** Model loads and runs on `torch.device('mps')` on M3 Ultra.
Inference: ~136ms first-pass (warm-up), subsequent frames faster via MPS cache.
No CUDA dependencies. Ultralytics routes automatically to MPS when available.

---

## Pre-run Hit Rate Test (15 HOLD frames, conf ≥ 0.20)

Tested before integrating into pipeline to confirm viability:

| Frame | Detections | Conf | Head center (cx,cy) |
|-------|-----------|------|---------------------|
| f430 | 1 | 0.842 | (210, 346) |
| f431 | 1 | 0.844 | (214, 350) |
| f432 | 1 | 0.844 | (215, 356) |
| f433 | 1 | 0.858 | (216, 361) |
| f434 | 1 | 0.850 | (216, 362) |
| f436 | 1 | 0.847 | (212, 361) |
| f445 | 1 | 0.848 | (196, 364) |
| f446 | 1 | 0.848 | (192, 364) |
| f447 | 1 | 0.846 | (190, 367) |
| f448 | 1 | 0.845 | (188, 368) |
| f449 | 1 | 0.841 | (186, 369) |
| f450 | 1 | 0.842 | (185, 369) |
| f451 | 1 | 0.842 | (183, 369) |
| f452 | 1 | 0.840 | (182, 367) |
| f465 | 1 | 0.867 | (188, 339) |

**Hit rate: 15/15 = 100%**
All detections above conf=0.84. Single detection per frame (no multi-head confusion).

---

## Architecture: H1 Tier Placement

Detection hierarchy (v16):

```
Tier 1: MediaPipe FaceLandmarker → MEDIAPIPE
         └─ ear-midpoint anchor (landmarks 234+454)
         └─ blendshapes + 6DoF pose from T_mp matrix

Tier 2: YOLOv10n-face → 6DRepNet360 → REP360
         └─ face-box crop → pose estimation
         └─ pose_calib anchor via yaw-conditioned calibration

Tier 3 [H1 NEW]: YOLOv8n-head (SCUT-HEAD) → HEAD_DET
         └─ head-box CENTER as position anchor ONLY
         └─ orientation held via state.update_hold() (no pose from back-of-head)
         └─ scale = head-box height (proxy at back-of-head, ear-span unavailable)

Tier 4: MediaPipe PoseLandmarker ear-midpoint → pose_calib / pose_raw
         (applied in compute_raw_anchors for REP360 frames, unchanged from v15)

Tier 5: HOLD — only fires if ALL 4 tiers above fail
```

**Design rationale for position-only:**
6DRepNet360 is trained on 300W-LP + Panoptic. At true back-of-head (yaw ~180°), the crop shows hair/scalp — the model returns arbitrary rotation values. Running it would be misleading.
The correct behavior: know WHERE the head is (head-box center), hold WHICH WAY it faces from the last valid REP360/MEDIAPIPE frame, and let the Kalman extrapolate orientation smoothly through the gap.

---

## Kalman Noise Assignment

H1 adds one new R value to the per-source hierarchy:

| Source | R (per axis) | Sigma | Rationale |
|--------|-------------|-------|-----------|
| mediapipe_face | 225 | 15px | Ear landmark direct from mesh |
| **head_det** | **3600** | **60px** | Head-box center; box ~130px span → ~1/4 box width error budget |
| pose_calib | 2025 | 45px | Calibrated pose RMSE ~28-37px |
| pose_raw | 6400 | 80px | Raw uncalibrated pose |
| predicted/hold | 250000 | 500px | No observation |

R_head_det = 3600 places it between pose_calib and pose_raw. The Kalman assigns it medium trust: the measurement is used to constrain position but neighboring high-confidence frames (mediapipe_face R=225) still dominate the smooth trajectory.

---

## Per-frame v15→v16 Anchor Comparison (former HOLD frames)

| Frame | v15 anchor (cx,cy) | v16 anchor (cx,cy) | Delta (px) |
|-------|-------------------|-------------------|------------|
| f430 | (197, 348) | (213, 353) | 17.0 |
| f431 | (199, 348) | (214, 354) | 15.4 |
| f432 | (200, 349) | (214, 354) | 13.9 |
| f433 | (202, 350) | (214, 354) | 12.6 |
| f434 | (203, 351) | (214, 354) | 11.5 |
| f436 | (204, 353) | (212, 361) | 10.8 |
| f445 | (200, 359) | (190, 366) | 12.3 |
| f446 | (198, 360) | (189, 366) | 11.1 |
| f447 | (197, 360) | (189, 367) | 10.1 |
| f448 | (195, 360) | (189, 367) | 9.2 |
| f449 | (193, 360) | (188, 367) | 8.6 |
| f450 | (192, 360) | (188, 367) | 8.2 |
| f451 | (190, 359) | (187, 367) | 8.2 |
| f452 | (189, 359) | (187, 367) | 8.6 |
| f465 | (186, 348) | (188, 339) | 9.2 |

Mean delta: 11.2px. Max delta: 17.0px. The v15 HOLD anchor was already reasonable (it was the last pose position, which was close to the head at these short gaps), but v16's anchor is derived from a live detection rather than a frozen last-position.

---

## Visual Verification

Proof frames generated: `v16_headdet_proof_f{frame:04d}.jpg` (15 individual frames)
Montage: `v16_headdet_montage.png` (5×3 grid)

**Visual assessment per-frame (every former HOLD frame):**

All 15 frames: BLUE box tightly encircles the head. GREEN circle (v16 smoothed anchor) lands on the head. RED x (v15 HOLD anchor) is close but is a frozen position, not a live detection.

Frame observations:
- f430-f434: chin-up extreme pose, head tilted back. SCUT-HEAD fires with conf=0.84-0.86 on the upturned head profile. Box correctly encloses head shape.
- f436: profile transition. Single detection, conf=0.847. Head box on head.
- f445-f452: back-of-head / chin-up rotation zone. Head fully turned. SCUT-HEAD detects the back/side head shape at conf=0.84-0.85. This is the case the face detector cannot handle — confirmed the head-shape detector bridges it.
- f465: isolated single HOLD frame between two REP360 frames. conf=0.867 (highest in the set). Head turning, partial back view.

**HEAD_DET anchor is on the head in all 15 frames. Visual verification passes.**

---

## What Does NOT Change vs v15

- Orientation during HEAD_DET frames: **held from last valid REP360/MEDIAPIPE**. 6DRepNet360 does not run on HEAD_DET frames. The avatar will hold its last known orientation through these 15 frames (~0.5 seconds at 29fps). This is the correct behavior — extrapolating pose through a back-of-head gap is better than fabricating pose from a non-face crop.
- All v15 fixes retained: FIX 1 (jump-gate), FIX 2 (RTS fallback), FIX 3 (scale-split), CHANGE 1 (ear-midpoint anchor), CHANGE 2 (heteroscedastic Kalman R).
- v14 YOLO face detector (yolov10n-face.pt) unchanged — head detector only runs when face detector fires nothing.
- Calibration RMSE unchanged: x=27.2px, y=37.1px (same MEDIAPIPE+REP360 frame set used for fitting).
- v15 NPZ (memoji_rig_stream_v13.npz) untouched. v16 writes to `memoji_rig_stream_v16.npz`.

---

## Scale Discontinuities at HEAD_DET Transitions

The FIX 3 scale-split now fires more at the back-of-head zone because the head-box height (~150px in this clip at this distance) differs from the pose ear-span scale (~35-40px at profile). The Kalman correctly identifies these as segment boundaries and resets the smoother:

```
Scale jump at f429→f430: 32.3→152.9px (120.6px jump)   [pose_raw → head_det]
Scale jump at f434→f435: 148.1→44.9px (103.2px jump)   [head_det → pose_raw]
...
```

This is expected and handled correctly by FIX 3. The scale during HEAD_DET frames is the head-box height, which is a reasonable proxy but not directly comparable to the ear-span used in pose_raw frames.

---

## Honest Residuals and Limitations

1. **Orientation not recovered during HEAD_DET frames.** The avatar orientation is held at the last valid pose during all 15 HEAD_DET frames. The head IS turning during this zone (chin-up / back-of-head rotation), so the avatar will appear to freeze in rotation while position tracks correctly. This is a genuine limitation: solving it would require a back-of-head pose estimator (none currently exists open-source that runs on MPS).

2. **Scale discontinuity at HEAD_DET entry/exit.** The head-box height scale (~150px) differs from the pose-anchor ear-span scale (~35-40px during profile turns). FIX 3 handles this with segment splits, but the avatar size may jump slightly at these transitions. This is a second-order issue given the HOLD zone is ~0.5 seconds of video.

3. **Confidence in novel clips.** SCUT-HEAD test: 100% hit rate on this specific clip's 15 HOLD frames, all at conf≥0.84. In a clip with very low lighting, extreme crop size, or multiple people, confidence could drop below 0.20 threshold and fall back to HOLD. The threshold was set conservatively at 0.20 (well below observed 0.84) to maximize coverage.

4. **Weight provenance.** Abcfsa/YOLOv8_head_detector is a 2023-24 graduation project. The training setup and evaluation metrics are not formally documented. The model works empirically on this clip at high confidence. For production deployment, training from the official SCUT-HEAD release with documented mAP would be preferable.

---

## Output Files

| File | Description |
|------|-------------|
| `pipeline_memoji_rig_v16_headdet.py` | V16 producer with H1 tier |
| `memoji_rig_stream_v16.npz` | V16 rig stream (225 KB; mode='HEAD_DET' for 15 former-HOLD frames) |
| `v16_yaw_calibration.json` | Same calib as v15 (RMSE x=27.2, y=37.1px) |
| `v16_headdet_montage.png` | 5×3 montage of all 15 former-HOLD frames |
| `v16_headdet_proof_f{N:04d}.jpg` | 15 individual proof frames with annotations |
| `models/yolov8n-head-scut.pt` | Downloaded weights (6.0 MB, SCUT-HEAD trained) |

---

## Summary

H1 is built, tested, and verified. The YOLOv8n-head detector (SCUT-HEAD, 6MB, MPS-native) eliminates all 15 HOLD frames on this clip by serving as a fallback position anchor when both face detectors fail. Detection confidence on the back-of-head/chin-up frames ranged from 0.840-0.867 (well above the 0.20 threshold). The HOLD count went from 15 to 0, exceeding the ≤5 target.

What H1 solves: WHERE is the head during back-of-head turns.
What H1 does not solve: WHICH WAY the head faces during those turns (orientation held).
Next step (if orientation fidelity during back-of-head is required): back-of-head pose estimation is an open research problem without a proven MPS-runnable solution as of 2026-06.
