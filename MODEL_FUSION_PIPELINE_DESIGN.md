# Model-Fusion Pipeline Design for Head/Face Tracking + Avatar

Date: 2026-06-14
Workspace: `.`

## Scope and Evidence

This design is grounded in:

- `RESEARCH_INNOVATE_TRACKING.md`
- `notes_grid_confirm.md`
- `notes_live_causal.md`
- `NOVELTY_TRANSFER_ASSESSMENT.md`
- `audio_prior_correlation_study.md`
- `notes_h1_headdet.md`
- Current streams: `memoji_rig_stream_v13.npz`, `memoji_rig_stream_v16.npz`, `live_causal_v1_stream.npz`

Current measured state:

- v15: `MEDIAPIPE=376`, `REP360=456`, `HOLD=15`.
- v16 H1: `MEDIAPIPE=376`, `REP360=456`, `HEAD_DET=15`, `HOLD=0`.
- v16 H1 solved head position for the former HOLD frames, not orientation.
- Causal IMM v1 matched offline v15 within 50 px on all 847 frames, mean position delta 7.7 px, but ran 24.9 fps on a 29.0 fps source.
- Audio prior is killed for near-term build: no prosody feature cleared the practical correlation gate of `|r| >= 0.3`; best was RMS energy vs head velocity at `r=0.241`.

Core thesis: do not average model outputs as if every model observes the same thing. Treat every model as a sensor over a subset of state dimensions: position, scale, orientation, expression, identity, and confidence. Fuse only the dimensions a model can actually observe.

## 1. Strengths and Weaknesses by Model

| Model / lane | Strongest signal | Honest weaknesses | Best role in fusion |
|---|---|---|---|
| MediaPipe FaceLandmarker | Dense face mesh, 478 3D landmarks, blendshapes, facial transform matrix. In this repo, direct face-ear midpoint is the highest-trust anchor (`sigma ~= 15 px`). | Face-visible model. It is strongest near frontal and degrades/drops at profile, heavy pitch, occlusion, and back-of-head. Can produce jumpy anchors at reacquisition. No multi-person identity protection by itself. | Primary source for face-visible position, orientation, expression, and avatar blendshapes. Use only while face confidence and landmark geometry are coherent. |
| YOLOv10n-face | Fast face box. Extends coverage into profile when MediaPipe face drops and enough face is still visible. Provides crop for 6DRepNet360. | Requires visible face. Box center is not a stable anatomical anchor. No orientation or expression. Can jitter and can choose wrong face in multi-person/mirror/TV scenes unless associated temporally. | Face-visible fallback detector and crop generator for 6DRepNet360. It should not directly drive final anchor except as a crop/quality signal. |
| 6DRepNet360 | Full-range rotation estimator from a crop; current strongest profile/head-pose lane when a face/head crop contains useful pose cues. Runs on MPS in current stack. | Needs a usable crop. At true back-of-head, pose is underconstrained and can be plausible-but-wrong. Needs yaw calibration for anchor alignment. No position by itself. No expression. | Primary orientation source for profile and non-MediaPipe face frames. Fuse orientation only when crop quality and yaw observability are acceptable. |
| YOLOv8n-head / SCUT H1 | Detects head shape, including side/back/occluded heads. In v16, turned 15 HOLD frames into 15 HEAD_DET frames with 0 HOLD. | Position only. No orientation, expression, or anatomical face point. Head-box scale differs from ear-span scale and can introduce scale discontinuities. Weight provenance needs production audit. | Back/profile position oracle. Updates `cx, cy, scale` with medium trust (`sigma ~= 60 px` in v16). Must not update orientation. |
| MediaPipe Pose | Body/ear/shoulder landmarks; can provide approximate head position when face lanes fail. Official pose model outputs 33 3D body landmarks including ears. | Coarse for head. Ears may be occluded or hallucinated. CPU bottleneck in live run (`21.4 ms/frame`). Position only for this task; no reliable head orientation/expression. | Last-resort position anchor, calibration support, and sanity check against head detector. Run on demand or alternate frames for live. |
| Heteroscedastic Kalman / IMM | Smooths across noisy sensors, source-aware `R`, constant-velocity plus maneuver model for causal output. Existing v15/v16 values: face 15 px, pose_calib 45 px, head_det 60 px, pose_raw 80 px, hold 500 px. | Garbage-in, garbage-out. It does not create missing visual evidence. If source covariance is wrong, it can hide errors or smear transitions. | State estimator and uncertainty propagator, not an oracle. Should output uncertainty and observed dimensions per frame. |
| OpenSeeFace candidate | Stateful avatar-oriented face tracker with stable landmarks across adverse face-visible conditions and wide head poses. | Still face-visible. Not true back-of-head. Outputs 68 landmarks, not 478; blendshape mapping is not drop-in. Integration cost. | Parallel stateful face lane to bridge MediaPipe dropout, profile boundary jitter, and short occlusions. Good live candidate if MediaPipe CPU cost is too high. |
| SAM2 candidate | Video object segmentation with memory. Can track a prompted head/person mask across frames, including difficult visual continuity cases. | Segmentation, not pose. Needs prompt/init and can drift to hair/body/background. Heavy integration; live MPS performance must be measured. | Offline or high-latency position/scale oracle and cross-check for head boxes. Useful for mask centroid, head silhouette, and occlusion flags. |
| FLAME / 3DMM tracker candidate | Geometry-consistent skull/face model; stable canonical head center; can optimize pose/expression against landmarks/masks and provide render residuals. | Heavy/offline, initialization-sensitive, CUDA/toolchain risk depending implementation. Still cannot observe true back-of-head orientation without asymmetric visual evidence. Licensing/assets need review. | Offline verifier/teacher, render-and-compare scorer, and geometry-consistent pseudo-label generator. Not first live path. |
| Optical flow candidate | Fast temporal bridge for landmarks/box points. Good for 5-15 frame detector gaps and smooth continuity. | Drifts, fails under large motion, occlusion, blur, and appearance change. No semantic understanding. No orientation/expression except propagated old values. | Short-horizon bridge with age-inflating uncertainty and forward-backward error gating. Never a long-gap primary tracker. |

## 2. Weakness x Coverage Matrix

| Weakness / failure mode | Covered by | How coverage works | Remaining gap |
|---|---|---|---|
| MediaPipe drops at profile/back-of-head | YOLOv10n-face + 6DRepNet360; YOLOv8n-head; OpenSeeFace; optical flow | Face detector covers partial profile; head detector covers head-shape position; OpenSeeFace/flow bridge boundary frames. | Back-of-head orientation and expression remain unobserved. |
| YOLO face needs visible face | YOLOv8n-head, SAM2, MediaPipe Pose | Head box/mask/body ears can keep position when face is gone. | No face expression; no trustworthy pose if only back of head is visible. |
| 6DRepNet360 needs valid crop | YOLOv10n-face, YOLOv8n-head, SAM2 crop/mask | Better crop selection and mask constraints reduce bad inputs. | Back-of-head crop still does not contain enough orientation evidence. Do not run/update orientation there. |
| 6DRepNet360 plausible-but-wrong at true back | Hard observability gate, orientation HOLD/predict, FLAME/render verifier | Mark orientation unobserved; propagate with rising uncertainty; reject pose updates from head-det-only frames. | True back-of-head yaw/pitch/roll is not covered by any current open MPS lane. |
| Head detector gives position only | 6DRepNet360, MediaPipe Face, OpenSeeFace for face-visible orientation | Update position and scale from head detector; keep orientation from last valid face/pose lane. | During full back view, orientation is only prior/prediction, not measurement. |
| Head-det scale differs from ear-span scale | Scale-specific covariance, source-specific scale model, SAM2 mask, segment reset | Do not mix bbox height and ear-span as identical measurements. Keep `scale_source` and convert with calibration. | Absolute depth/scale remains monocular and camera-dependent. |
| MediaPipe Pose is coarse and CPU-heavy | FaceLandmarker, YOLO-head, OpenSeeFace; run pose on demand | Use pose only when needed or every N frames; let IMM predict skipped frames. | If all face/head signals fail, pose may be the only weak cue. |
| Mixed anchor references cause jumps | Ear-midpoint unification already fixed v15; FLAME canonical center can improve | Use one anatomical/semantic reference per output dimension. Track source-specific offsets explicitly. | Head-box center is still a different reference than ear midpoint unless calibrated. |
| Kalman can smear wrong measurements | Heteroscedastic R, innovation gating, source hysteresis, segment boundaries | Low-trust sources pull less; large Mahalanobis innovations are rejected/downweighted. | If a wrong source is confidently wrong and no other evidence exists, fusion can still fail. |
| Fast motion / camera zoom | IMM CV+NCA, optical flow, scale-reset, higher process noise | Maneuver model handles sudden acceleration; flow bridges detector cadence; scale reset prevents RTS smear. | Severe motion blur can break all visual measurements. |
| Short occlusion | Optical flow, OpenSeeFace state, SAM2 memory, Kalman prediction | Continue local track with uncertainty increasing by age. | Long occlusion with no head/body visibility becomes prediction only. |
| Facial expression lost at profile/back | MediaPipe when visible; OpenSeeFace mouth landmarks as partial backup; decay/hold | Expression is face-observed only. Decay or hold with confidence flag. | Back-of-head expression is fully uncovered. Do not infer from audio; audio prior failed. |
| Multi-person or mirror/TV false detection | ByteTrack/SORT-style association, SAM2 track identity, IoU continuity, face/head size priors | Associate detections to existing track before accepting. | ReID is not implemented; current stack is single-person-biased. |
| Motion blur / low light | OpenSeeFace may help; SAM2/flow sometimes help; Kalman prediction | Redundant visual lanes may not fail on same frame. | If image evidence is gone, only prediction remains. |
| Camera intrinsics / depth ambiguity | FLAME/camera fit, bbox/ear-span scale calibration, known focal calibration | Better model explains perspective and avatar scale. | Monocular absolute depth is not solved without stronger calibration or depth sensor. |
| Audio as motion prior | None near-term | Gate study killed it for this clip. | Reopen only after multi-clip corpus with held-out prediction gain. |

Uncovered gaps to state explicitly:

1. True back-of-head orientation is not solved. Head detector and SAM2 can locate the head; they cannot know the face-facing direction with high confidence when no asymmetric facial evidence is visible.
2. Back/profile expression is not solved. Blendshapes should decay/hold and be marked low confidence.
3. Full occlusion is not solved. Kalman/IMM gives a prediction, not a measurement.
4. Multi-person identity is not solved by the current single-person cascade. It needs track association before production claims.
5. Absolute 3D depth/scale is not solved by monocular fusion; current scale is a calibrated 2D proxy.

## 3. Fusion Architecture Options

| Option | Value | Risk | Verdict |
|---|---|---|---|
| Cascade + heteroscedastic Kalman | Simple, already working. v15/v16 evidence is strong on this clip. Cheap to run. | Ordered decisions discard secondary evidence; mode switches can still be brittle; orientation/position can be conflated. | Keep as baseline and live fallback. |
| Confidence-weighted multi-model fusion | Uses all available observations with covariances. Reduces dependence on any single detector. | Bad if confidences are uncalibrated or if unobserved dimensions are accidentally averaged. | Best core design, but only with hard observability masks. |
| Parallel vote + arbitration | Good outlier rejection. If MP, head-det, pose, and flow disagree, pick the source that is observable and temporally consistent. | Requires careful thresholds and source hysteresis. | Add around the current cascade. |
| Learned gating network | Can learn when to trust each model from features: yaw, bbox conf, landmark residual, blur, flow error, source age. | One clip is not enough. High overfit risk. Needs labeled corpus and held-out clips. | Later. Treat as a research hypothesis, not v17 default. |
| Render-and-compare cross-check | Uses avatar/FLAME projection residuals or mask agreement to catch wrong pose/anchor. | Slow and requires geometry/camera setup. | Excellent offline verifier and teacher; not first live path. |
| Complementary redundancy | Separate lanes for face landmarks, face pose, head position, body pose, temporal flow, segmentation. Fuse by observable dimension. | More plumbing and bookkeeping. | Recommended architecture. |

## Recommended Setup: Factorized Complementary-Redundancy Fusion

### State

Maintain a factorized state:

```text
position:    cx, cy, vx, vy
scale:       head_scale, scale_velocity
orientation: quaternion or yaw/pitch/roll + angular velocity
expression:  ARKit blendshape vector + expression confidence
quality:     per-dimension sigma/confidence, source, observed_dims, failure_reason
```

Do not use one global `mode` as the only truth. A single frame can be:

- position observed by `head_det`
- scale observed by `head_det_bbox_height`
- orientation unobserved and predicted
- expression held/decayed

### Measurement Lanes

| Lane | Position update | Scale update | Orientation update | Expression update |
|---|---|---|---|---|
| MediaPipe Face | Yes, high trust, face-ear midpoint | Yes, ear-span/mesh | Yes, if landmark geometry coherent | Yes, primary |
| YOLO face + 6DRepNet360 | Via pose_calib/body anchor, medium trust | Face box / calibrated proxy | Yes, if crop quality and yaw observability pass | No |
| YOLO head | Yes, medium trust, head box center | Yes, bbox height with separate source model | No | No |
| MediaPipe Pose | Yes, medium/low trust depending ears/shoulders | Weak | No, except optional very coarse body-facing prior with huge R | No |
| OpenSeeFace | Yes, medium/high when tracker confident | Yes, landmark span | Yes, medium if face-visible | Partial mouth/face only if mapping is validated |
| Optical flow | Yes, propagated with age-inflating R | Maybe | Only propagate prior, no new measurement | Only propagate prior |
| SAM2 | Yes, mask centroid/ellipse, medium trust | Yes, mask/bbox | No | No |
| FLAME/3DMM | Yes, canonical center, offline | Yes | Yes when landmarks/mask constrain it | Yes when face visible |

### Confidence and Uncertainty

Use source base sigmas from the current measured pipeline, then inflate them by confidence and failure-mode features:

```text
sigma_pos_base:
  mediapipe_face: 15 px
  pose_calib:     45 px
  head_det:       60 px
  pose_raw:       80 px
  predicted:     500 px
```

Proposed additions:

```text
open_seeface: 20-50 px, based on tracker confidence and landmark residual
optical_flow: 20 px + 8-12 px/frame age + forward-backward error
sam2_mask:    25-60 px, based on mask stability, IoU, and head/person separation
```

For each model and state dimension:

```text
if dimension is not observed:
    no measurement update for that dimension
else:
    R = sigma_base^2
    R *= confidence_inflation
    R *= yaw_or_occlusion_inflation
    R *= blur_crop_quality_inflation
    reject/downweight if Mahalanobis innovation is too large
```

Critical rule: head detector, SAM2, and body pose must not update orientation. They can update position/scale only.

### Fusion Loop

1. Run cheap/primary lanes: MediaPipe Face, optional OpenSeeFace state, pose-on-demand or alternate-frame pose.
2. If face confidence is low, run YOLO face + 6DRepNet360.
3. If face lane fails or yaw/crop quality is low, run YOLO head. For offline/failure probes, also run SAM2 from the best recent head prompt.
4. Build per-lane measurements with `observed_dims`, covariance, and source metadata.
5. Apply arbitration:
   - Identity association first: IoU/track continuity before accepting a detection.
   - Innovation gate: reject measurements that require impossible jumps unless another source agrees.
   - Source hysteresis: avoid one-frame mode flapping.
   - Cross-source checks: head-det center should be near pose/head prediction; face crop should lie inside head mask/box.
6. Update state:
   - Position/scale: heteroscedastic IMM Kalman.
   - Orientation: quaternion/SO(3) filter using MediaPipe/6DRepNet/OpenSeeFace/FLAME only.
   - Expression: MediaPipe primary; OpenSeeFace/FLAME only after mapping validation; otherwise hold/decay.
7. Emit final avatar stream with explicit confidence:
   - `pos_sigma_px`
   - `rot_sigma_deg`
   - `expr_conf`
   - `position_source`
   - `orientation_source`
   - `expression_source`
   - `observed_dims`
   - `failure_reason`

### Recommended Build Path

v17 target:

1. Start from v16 H1, because it already eliminates HOLD position gaps on this clip.
2. Add factorized output fields: separate source/confidence for position, scale, orientation, and expression.
3. Port v16 H1 into the causal IMM path so live and offline have the same model coverage.
4. Add optical-flow bridge for short gaps, with age-limited uncertainty.
5. Add OpenSeeFace as a parallel probe, not a replacement, and measure whether it reduces transition jitter.
6. Add render/mask cross-check offline: SAM2 first for position/scale; FLAME later as verifier/teacher if integration cost is acceptable.
7. Do not add learned gating until there is a labeled multi-clip eval set.

## 4. Ranked Testable Hypotheses

| Rank | Hypothesis | Metric | Kill condition |
|---|---|---|---|
| 1 | Factorized v16+causal fusion beats current cascade on failure frames because position, orientation, and expression are no longer conflated. | On manually labeled failure frames: position mean/P90 error, orientation snap count, expression confidence correctness. Compare v16 cascade vs factorized causal v17. | Kill if P90 position improves <10 px or stable MEDIAPIPE frames degrade >5 px mean, or if orientation confidence flags do not match visible failure cases. |
| 2 | Adding YOLO-head to the causal IMM path eliminates live HOLD frames without quality regression. | HOLD count, HEAD_DET count, mean/P90 delta vs offline v16, max jump, FPS. | Kill if HOLD remains >5 on this clip, P90 delta vs offline v16 >25 px, or achieved fps drops below source by >20% after pose-on-demand optimization. |
| 3 | OpenSeeFace plus optical flow reduces seam jitter at MediaPipe/REP360/profile transitions. | Transition-window jerk, max frame-to-frame anchor jump, manual anchor error on f417-f484/f521-f575/f694/f758 style zones. | Kill if transition P90 jump reduction <25%, flow/OSF drift exceeds 25 px for >3 consecutive frames, or false continuity hides true detector loss. |
| 4 | SAM2 mask fusion improves back-of-head position/scale over YOLO-head-only. | Former HOLD frames and back/profile frames: mask/head-box center error, scale jump at entry/exit, silhouette IoU, visual avatar size stability. | Kill if center P90 improves <10 px over YOLO-head or scale jumps remain >50 px after source-specific scale calibration, or runtime is unacceptable for the target mode. |
| 5 | Orientation fusion with hard observability gates reduces pose snaps without inventing back-view pose. | Quaternion angular velocity spikes, render residual, manual coarse yaw labels, number of frames marked `orientation_unobserved`. | Kill if angular spike count does not drop by >20% on profile/occlusion frames, or if back-of-head frames are assigned false high orientation confidence. |
| 6 | Render-and-compare arbitration catches catastrophic wrong-pose frames. | True/false positive rate of flags against visual review, reprojection residual distribution, unique failures caught. | Kill if false positive rate >5% on clean face-visible frames, or if it catches no unique failures beyond innovation gating. |
| 7 | Learned gating improves deterministic fusion only after a multi-clip labeled corpus exists. | Held-out clip P90 position/orientation error, calibration of predicted uncertainties, failure-mode recall. | Kill if improvement over deterministic gates is <10% on held-out clips or if performance collapses on unseen lighting/profile/body motion. |
| 8 | FLAME/3DMM can serve as offline teacher/verifier for geometry-consistent anchors. | Fit success rate, runtime/frame, reprojection/mask residual, anchor consistency across yaw. | Kill if setup/runtime blocks iteration, if fit fails on profile/back clips, or if generated pseudo-labels disagree with manual labels by >25 px P90. |
| 9 | Audio prior remains a no-go unless a larger corpus proves predictive value. | Multi-clip held-out prediction gain for head velocity/pose; correlation above block-shuffle baseline; practical `|r| >= 0.3`. | Already killed for this clip. Reopen only if larger corpus clears the gate and reduces visual tracker error during occlusion. |

## 5. Bottom Line

The best next system is not "more models averaged together." It is a factorized, uncertainty-aware ensemble:

- Face-visible frames: MediaPipe Face and 6DRepNet/OpenSeeFace compete and cross-check orientation; MediaPipe owns expression.
- Profile frames: YOLO face + 6DRepNet and OpenSeeFace/flow bridge the face-landmark dropout.
- Back-of-head frames: YOLO head and optionally SAM2 own position/scale; orientation is predicted/held with rising uncertainty; expression decays.
- Fast motion: IMM, optical flow, and source-specific covariances keep continuity without pretending predictions are measurements.
- Offline verification: SAM2/FLAME/render residuals catch false positives and can generate future training labels.

This should improve robustness at the ensemble level because each model covers a different observability hole. The major unsolved hole remains true back-of-head orientation; no current lane in this stack directly observes it.

## External Primary Sources Used

- MediaPipe Face Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- 6DRepNet360: https://github.com/thohemp/6DRepNet360
- SCUT-HEAD dataset: https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
- YOLOv8 SCUT head detector: https://github.com/Abcfsa/YOLOv8_head_detector
- OpenSeeFace: https://github.com/emilianavt/OpenSeeFace
- SAM2: https://ai.meta.com/research/sam2/ and https://github.com/facebookresearch/sam2
- FLAME: https://flame.is.tue.mpg.de/
- FLAME video head tracker: https://github.com/philgras/video-head-tracker
- OpenCV Lucas-Kanade optical flow: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
