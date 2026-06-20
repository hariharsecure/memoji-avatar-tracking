# Real-Time Head-Pose & Blendshape Tracking for On-Device Avatar Puppeting

Monocular, real-time face / head-pose and **52-blendshape** tracking that drives
an avatar (virtual-character) overlay, with a **predict → observe → correct**
recursive state estimator and a multi-model fusion cascade. The whole stack
runs **on-device on Apple Silicon (Metal / MPS), CUDA-free** — an Edge-AI
constraint, not a cloud pipeline.

> **Scope of "multimodal" / "sensor".** This is a **single-modality, monocular
> RGB** system: there is one physical sensor (one camera) and **no** depth,
> IMU, audio, or multi-view input in the live path. "Sensor" below is used in
> the estimation sense — each *model head* (face mesh, face detector, head-pose
> regressor, head detector, body pose, optical flow) is treated as a virtual
> sensor over a subset of the state. The fusion is **multi-model / multi-cue**,
> not multi-sensor or multi-modal data fusion. (An audio modality *was*
> investigated as a head-motion prior and **killed** at the correlation gate —
> see "Honest limits".)

The problem this targets sits at the intersection of **uncertainty-aware state
estimation and machine learning**: fuse several heterogeneous, partially-reliable
**model outputs derived from one RGB stream** (a dense face mesh, a face
detector, a full-range head-pose regressor, a head detector, a body-pose
estimator, optical flow) into one continuous, uncertainty-aware estimate of a
head's pose and expression — and keep producing a defensible output even when the
face is not visible.

---

## Problem

Given a single RGB camera (a monocular talking-head clip, no depth sensor, no
multi-view rig), produce **every frame**:

- 6-DoF head pose (position + yaw / pitch / roll),
- a 52-coefficient ARKit-style blendshape vector (mouth, jaw, eyes, brows, etc.)
  for expression / lip-sync,
- a head-anchored avatar overlay,

robustly across the hard cases that break naive face trackers: **profile views,
extreme pitch, partial occlusion, and back-of-head frames** where the face
detector simply drops. The deliverable is a *never-drop* tracker: it never
silently emits a confident wrong answer, and it labels every quantity as
`verified`, `predicted`, or `mixed`.

This is a sensing + estimation problem before it is a rendering problem. The
core difficulty is that no single sensor observes the full state on every frame,
and the sensors that remain available in the hard cases (head box, body pose,
optical flow) observe *position* but not *orientation* or *expression*.

## Approach

### 1. Multi-model cascade, fused by observable dimension

Rather than averaging model outputs, each model is treated as a **virtual sensor
over a subset of the state** (position, scale, orientation, expression) — all
derived from the same single RGB stream — and only the dimensions a model can
actually observe are fused. The component model heads:

| Sensor | Observes | Role |
|---|---|---|
| **MediaPipe Face Landmarker** | dense 478-pt face mesh + 52 blendshapes + facial transform | primary, face-visible: position, orientation, expression |
| **YOLOv10-n face** | face bounding box | face-visible fallback detector + crop generator |
| **6DRepNet360** | full-range head rotation from a crop | primary orientation source at profile / non-frontal |
| **YOLOv8-n head** (SCUT-HEAD) | head bounding box incl. side / back | back-of-head **position** oracle (not orientation) |
| **MediaPipe Pose** | body / ears / shoulders (33 landmarks) | last-resort head-position prior; body→head kinematics |
| **Optical flow** | short-horizon point motion | bridges detector gaps with age-inflating uncertainty |

A hard **observability gate** enforces the key discipline: head detectors, body
pose, and flow may update position / scale but are **forbidden from updating
orientation**, because they carry no orientation evidence. This prevents the
classic failure of a back-of-head detector confidently asserting a wrong yaw.

### 2. Predict → observe → correct state estimator

The estimator (design in `PREDICT_CORRECT_DESIGN_v8.md`,
`MODEL_FUSION_PIPELINE_DESIGN.md`) is a confidence-weighted recursive filter over
a **factorized state** (position, scale, orientation, expression — each carrying
its own covariance):

- **Predict** each hidden quantity from correlated signals as a confidence-weighted
  prior — pose from a quaternion constant-angular-velocity model plus optical-flow
  head-anchor plus a soft body→head kinematic prior; expression from temporal
  blendshape dynamics.
- **Observe** with whichever sensors are valid on that frame.
- **Correct** with an **error-state Kalman filter on SO(3)** for rotation and a
  linear Kalman filter for position / velocity; expression with a robust filter
  over blendshapes.
- **Innovation gating** throughout: an observation that contradicts the prediction
  too strongly (large Mahalanobis innovation) is down-weighted or rejected rather
  than blindly snapped to.
- **Heteroscedastic, source-aware measurement noise**: each source has a base σ
  (face ≈ 15 px, calibrated pose ≈ 45 px, head-det ≈ 60 px, raw pose ≈ 80 px,
  prediction ≈ 500 px), inflated by per-frame confidence, yaw / occlusion, and
  crop-quality features.
- An earlier offline variant used a forward–backward (RTS) Kalman smoother; the
  current causal path uses an **Interacting Multiple Model (IMM)** filter
  (constant-velocity + maneuver model) so the same coverage runs live.

Every emitted quantity is tagged `verified` / `predicted` / `mixed`; a prediction
is **never** promoted to `verified` without a real observation.

### 3. On-device / Edge-AI inference

All detectors and the estimator run on Apple Silicon via PyTorch **MPS** and
CPU — no CUDA, no server round-trip. This is an explicit Edge-AI design
constraint: the survey in `3DGS_HEAD_AVATAR_ASSESSMENT.md` documents which
state-of-the-art head-avatar methods are *not* on-device-feasible (they require
CUDA rasterizers) and why this pipeline stays on the on-device path.

## Capabilities & Results

Measured on one real-world monocular clip (847 frames, 720×1280, 29 fps).
**These are single-clip engineering results, honestly framed — not benchmark or
generalization claims** (see "Claim discipline" below).

- **Per-frame coverage:** an estimate is produced for **847/847 frames**. A
  unified ear-midpoint anchor plus heteroscedastic Kalman puts the anchor on the
  head for every frame; adding the head detector reduced unanchored
  ("HOLD") frames from 15 to 0 in the head-detector variant.
- **Mesh / mask agreement (PRIMARY frames):** face-mesh convex-hull vs an
  independent YOLO face box gives **mean IoU 0.717** (frontal 0.732, profile
  0.654) over the 374 face-visible frames. (See `wireframe_accuracy_report.md`
  for the full distribution and the `[MEASURED]` vs `[PUBLISHED]` framing.)
- **Mode split:** 44.2% of frames are full per-vertex face mesh + live
  blendshapes; 55.8% are profile / back-of-head, served as pose-only canonical
  geometry with decayed (and explicitly low-confidence) blendshapes.
- **Expression tracking:** jaw / mouth blendshapes (e.g. `jawOpen`) track real
  open/close motion across consecutive frames on face-visible frames.
- **Throughput:** the causal IMM path runs ≈ 24.9 fps on the test hardware
  against a 29 fps source — near-real-time, with a documented path to close the
  gap (pose-on-alternate-frames). Source-rate live operation is *not* yet
  claimed.

### Honest limits (stated up front)

- **True back-of-head orientation is unsolved** by any sensor in this stack — no
  available on-device sensor observes face-facing direction with no facial
  evidence. Orientation there is a labeled prediction, never `verified`.
- **Single-person** cascade; multi-person / re-identification is not implemented.
- **Monocular absolute depth / scale** is a calibrated 2D proxy, not metric.
- An **audio→head-motion prior** was investigated and **killed** for this clip:
  no prosody feature cleared a practical `|r| ≥ 0.3` correlation gate (best was
  RMS energy vs head velocity at r = 0.241). Documented in
  `audio_prior_correlation_study.md` as a negative result.

## Claim discipline

This repo deliberately avoids novelty / "world-first" / "solved" language. The
components (uncertainty-weighted Kalman fusion, tracking-by-detection,
SO(3) filtering, IMM, occlusion prediction) all have prior art; the contribution
is a **pragmatic, uncertainty-aware integration** that is on-device-compatible
and locally validated on one clip. `NOVELTY_TRANSFER_ASSESSMENT.md` records the
explicit honest wording and the overclaims that are avoided.
`RANKED_NOVEL_TRACKING_IDEAS.md` separates "couldn't find exact prior art"
(testable open hypotheses) from "already done" — it does not assert verified
novelty.

## Methods & external references

YOLO face / head detection (Ultralytics; SCUT-HEAD for heads) · 6DRepNet360
full-range head-pose regression · MediaPipe Face Landmarker (478-pt mesh + 52
ARKit blendshapes) and Pose Landmarker · error-state Kalman filtering on SO(3) +
linear / IMM Kalman for position · optical flow (Lucas–Kanade) bridging.

- MediaPipe Face / Pose Landmarker — https://developers.google.com/mediapipe
- 6DRepNet360 — https://github.com/thohemp/6DRepNet360
- SCUT-HEAD dataset — https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
- Ultralytics YOLO — https://github.com/ultralytics/ultralytics

## Design docs

| Doc | Contents |
|---|---|
| [`PREDICT_CORRECT_DESIGN_v8.md`](PREDICT_CORRECT_DESIGN_v8.md) | The current predict→observe→correct state-estimator design, state variables, pre-registered metrics and kill conditions. |
| [`MODEL_FUSION_PIPELINE_DESIGN.md`](MODEL_FUSION_PIPELINE_DESIGN.md) | Full multi-model fusion architecture (each model head treated as a virtual sensor over a subset of the state — all from one RGB stream): per-model strengths/weaknesses, weakness×coverage matrix, factorized state, measurement lanes, ranked testable hypotheses. |
| [`RESEARCH_INNOVATE_TRACKING.md`](RESEARCH_INNOVATE_TRACKING.md) | Architecture critique of the per-frame-detect+smooth paradigm and its structural weaknesses; on-device (MPS) model survey. |
| [`RANKED_NOVEL_TRACKING_IDEAS.md`](RANKED_NOVEL_TRACKING_IDEAS.md) | Ranked research hypotheses with honest novelty assessment, MPS feasibility, PoC plans, and kill conditions. |
| [`NOVELTY_TRANSFER_ASSESSMENT.md`](NOVELTY_TRANSFER_ASSESSMENT.md) | Adversarial novelty / transfer assessment; recommended honest claim wording. |
| [`3DGS_HEAD_AVATAR_ASSESSMENT.md`](3DGS_HEAD_AVATAR_ASSESSMENT.md) | Literature review of 3D Gaussian Splatting head-avatar methods (CVPR/SIGGRAPH 2023–2026) assessed for on-device / MPS feasibility. |
| [`wireframe_accuracy_report.md`](wireframe_accuracy_report.md) | Measured accuracy report (mask IoU, pose, per-vertex residuals) with explicit measured-vs-published framing. |
| [`audio_prior_correlation_study.md`](audio_prior_correlation_study.md) | Negative-result study: audio prosody as a head-motion prior (killed at the correlation gate). |
| `notes_*.md` | Per-iteration engineering notes (method summaries, pre-registered metrics, honest-floor analyses). |

## Repository layout

```
pipeline_*.py            # tracking / estimation pipelines (see "current heads" below)
wireframe_overlay_v1.py  # wireframe diagnostic overlay
*_avatar_*.py            # avatar / mesh overlay renderers
*.md                     # design docs, measured reports, engineering notes
SETUP.md                 # how to fetch models, assets, and the 3DDFA_V2 dependency
requirements.txt         # Python dependencies (Python 3.11, MPS-compatible)
```

Current heads: `pipeline_predict_correct_v8.py` (estimator),
`pipeline_mesh_cascade_v7.py` (mesh/pose cascade), `wireframe_overlay_v1.py`
(diagnostics). Lower-numbered `pipeline_*` / `mesh_cascade_*` files are retained
as the iteration history behind the current design.

## Setup & run

See [`SETUP.md`](SETUP.md). In brief: create a Python 3.11 env,
`pip install -r requirements.txt`, download the model weights into `models/` and
assets into `assets/`, optionally clone `3DDFA_V2` into `_deps/` (only the
BFM-identity experiment needs it), drop a monocular clip as `input_clip.mov`, and
run a `pipeline_*.py` script. Model weights, assets, the vendored dependency, and
media are **not** committed (see `.gitignore`).

## License / third-party

This repository is the author's own pipeline code and design docs. It does
**not** redistribute third-party model weights or the 3DDFA_V2 project (those
carry their own licenses; the Basel Face Model used by the optional BFM
experiment is academic-use-only). Fetch them per `SETUP.md`.

---

**Author:** Harihar Thapa
