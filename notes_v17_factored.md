# v17 Factorized Avatar Stream — Build Notes

Date: 2026-06-14
Based on: MODEL_FUSION_PIPELINE_DESIGN.md
Builds on: pipeline_memoji_rig_v16_headdet.py

---

## What Was Built

`pipeline_memoji_rig_v17_factored.py` refactors the output representation of the v16 pipeline
from a single flat record (mode + anchor_source + yaw/pitch/roll) into a FACTORIZED stream
where each avatar state dimension carries its own source tag, confidence, and sigma:

| Dimension   | Fields added                                              |
|-------------|-----------------------------------------------------------|
| position    | pos_source, pos_sigma_px                                  |
| scale       | scale_source                                              |
| orientation | orient_observed (bool), orient_source, rot_sigma_deg      |
| expression  | expr_conf, expr_source                                    |
| failure     | failure_reason (per frame)                                |

All v16/v15/v14 detector logic is UNCHANGED: same cascade, same Kalman, same jump-gate, same
scale-segment split, same heteroscedastic R values. This is an output representation refactor.

---

## Verification Results (Run on input_clip.mov, 847 frames)

Frame counts: MEDIAPIPE=376, REP360=456, HEAD_DET=15, HOLD=0

Critical rule check (both reviewers required):
- HEAD_DET all orient_observed=False: PASS (15/15 frames)
- HEAD_DET all pos_source=head_det:   PASS (15/15 frames)
- MEDIAPIPE all orient_observed=True: PASS (376/376 frames)
- REP360 all orient_observed=True:    PASS (456/456 frames)
- orient_observed=False count == HEAD_DET count: PASS (15 == 15)
- OVERALL verification: PASS

---

## HEAD_DET Frame Detail — Honest Orientation Reporting

The 15 back-of-head frames now explicitly declare:
  - pos_source = 'head_det'    (position comes from YOLOv8n-head bbox center, sigma=60px)
  - orient_observed = False    (NEVER faked — head detector sees no face)
  - orient_source = 'held_head_det'  (last valid REP360/MEDIAPIPE value held in transform)
  - rot_sigma_deg RISES by 3°/frame while unobserved
  - failure_reason = 'back_of_head_no_orient'

rot_sigma values on HEAD_DET frames:
  Zone 1 (f430-f434): 8→11→14→17→20° (5 frames, +12° total)
  f435 is REP360 (orient_observed=True, sigma reset to 5°)
  f436 HEAD_DET: sigma = 5+3 = 8° (correct — inherits from the REP360 interrupt)
  Zone 2 (f445-f452): 8→11→14→17→20→23→26→29° (8 frames, +21° total)
  f465 HEAD_DET: sigma = 8° (single-frame zone after f464 REP360)

The warning flagged in verification ("not monotonically rising in zone 430-436") is a true
negative: f435 is a REP360 frame that interrupts the zone. The sigma reset from 20° to 5° at
f435 then rises again to 8° at f436 is CORRECT behavior, not a bug. The verifier was checking
the HEAD_DET-only frames as a contiguous zone including non-HEAD_DET frames in the range.

Yaw values on HEAD_DET frames are the last-valid held value:
  Zone 1: held at 81.1° (last REP360 yaw before back-of-head)
  Zone 2: held at 159.4° (last REP360 yaw at ~true back-of-head)
  f465: held at 50.8° (single-frame, last REP360 yaw)

These yaw values in the NPZ are NOT fresh measurements. The orient_observed=False flag is the
machine-readable signal that downstream consumers must check before trusting the orientation.

---

## What This Step Does NOT Claim

1. No quality improvement over v16 in position tracking. The Kalman, cascade, and R values
   are identical. The smoothed head_center_px output is numerically the same as v16.

2. The held yaw/pitch/roll values on HEAD_DET frames are not improved estimates. They are the
   same held values v16 stored, just now explicitly flagged as unobserved.

3. Back-of-head orientation remains the single uncovered gap, as documented in both design
   docs. No model in the current MPS-runnable stack provides correct orientation at yaw~180°.
   The factorized representation makes this honest — it was previously implicit.

4. This is the FOUNDATION step. The fusion quality gains (SAM2 dual-position, rotation Kalman
   in state, cross-agreement check) are subsequent steps that build on this factorized API.

---

## NPZ Output Fields (v17 additions)

New fields in memoji_rig_stream_v17.npz (all v16 fields retained for backward compat):

  pos_source        — per-frame str: which model provided position
                      values: 'mp_ear_midpoint', 'rep360_calib', 'head_det',
                              'hold_predicted', 'mp_fallback_pose'
  pos_sigma_px      — float32: position uncertainty (px), one sigma
                      values: 15px (mp), 45px (rep360), 60px (head_det), 500px (hold)
  scale_source      — per-frame str: which model provided scale
  orient_observed   — bool: True if orientation was measured this frame
                      False on HEAD_DET and HOLD frames
  orient_source     — str: 'mediapipe', 'rep360', 'held_head_det', 'held_hold', 'none'
  rot_sigma_deg     — float32: orientation uncertainty (degrees), one sigma
                      rising when orient_observed=False
  expr_conf         — float32: expression confidence (1.0=fresh MP, decays when absent)
  expr_source       — str: 'mediapipe', 'hold_decay', 'none'
  frames_since_orient — int32: frames elapsed since last valid orientation measurement
  failure_reason    — str: '' on clean frames; explanation on degraded frames

---

## Files Produced

  pipeline_memoji_rig_v17_factored.py   — this pipeline
  memoji_rig_stream_v17.npz             — factorized stream (230 KB)
  v17_factored_montage.png              — 4-panel diagnostic:
    panel 0: rot_sigma_deg (green=observed, red=unobserved, orange lines=HEAD_DET)
    panel 1: yaw values (green dots=measured, red x=HELD/unobserved)
    panel 2: pos_sigma_px by source (color coded)
    panel 3: expr_conf over time
  notes_v17_factored.md                 — this file

---

## Sigma Model

Orientation sigma per-source (deterministic, set from design constants):
  MEDIAPIPE frame:   2.0°  (facial_transformation_matrix, high trust)
  REP360 frame:      5.0°  (6DRepNet360 empirical on 300W-LP+Panoptic)
  HEAD_DET/HOLD:     prev_sigma + 3.0°/frame, capped at 180°

These are design priors, not empirically measured per-frame uncertainty from a HHP-Net-style
heteroscedastic pose model. A future step (v17 H3 in the design docs) would replace the
fixed per-source sigma with per-frame uncertainty from HHP-Net or similar.

---

## Next Steps (from design docs)

Step 2 (H3, Layer 4): Add (yaw, pitch, roll) to the Kalman state vector. Currently yaw/pitch/roll
are stored raw per-frame but not filtered. A rotation-state Kalman would smooth the
frame-to-frame jitter in the REP360 zone. This requires careful angle wrapping at ±180°.

Step 3 (H2, Layer 3): SAM2-tiny as a parallel position oracle on HEAD_DET frames. This would
provide a second independent position measurement alongside head_det, weighted by R_sam2=4900.
The factorized stream makes this a clean add: pos_source can be 'sam2_fusion' with its own sigma.

Step 4 (Layer 2): Cross-agreement check (MP vs YOLO) to detect hallucinated MP landmarks.
The factorized pos_sigma is the natural place to inflate on disagreement.
