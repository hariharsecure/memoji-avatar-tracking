# notes_v17_fusion.md

**Pipeline:** `pipeline_v17_fusion.py`
**Date:** 2026-06-14
**Source clip:** `input_clip.mov` (847 frames, 720x1280, 29fps)
**Base:** v17 step-2 / `pipeline_live_causal_v2.py`

## Built Layers

1. Cross-agreement check: MediaPipe face detections are checked against a YOLOv8n-head box and fresh pose anchor when available. Strong disagreement rejects MediaPipe before it can update position, orientation, or expression.
2. Optical-flow bridge: short all-detector gaps use sparse Lucas-Kanade propagation from the last real anchor, capped at 5 frames with rising position sigma.
3. Rotation Kalman: yaw/pitch/roll are smoothed with a per-axis causal filter updated only by MediaPipe and 6DRepNet360 orientation measurements.

## Measured Benchmark

| Metric | Value |
|---|---:|
| Frames processed | 847 |
| Achieved FPS | 35.1 |
| Mean frame time | 27.5 ms |
| P90 frame time | 49.1 ms |
| HOLD frames | 0 |
| FLOW_BRIDGE frames in real run | 0 |
| Lock rate vs v17 offline (<=50px) | 98.5% |
| Mean delta vs v17 offline | 8.3 px |
| P90 delta vs v17 offline | 18.7 px |

## Mode Breakdown

| Mode | Frames | % |
|---|---:|---:|
| MEDIAPIPE | 374 | 44.2% |
| REP360 | 457 | 54.0% |
| HEAD_DET | 16 | 1.9% |
| FLOW_BRIDGE | 0 | 0.0% |
| HOLD | 0 | 0.0% |

## Fusion Verification

- Cross-agreement planted bad MP: PASS; rejected=True reason=mp_outside_head_box+mp_head_center_disagree(713.8>208.5); planted distance=713.8px.
- Optical-flow synthetic gap: PASS; bridged 5/5 frames; mean error=2.0px; p90=3.6px.
- Rotation smoothing: PASS; angular jitter RMS raw=21.980 deg, smooth=9.651 deg; reduction=56.1%.

## Factorized Output Verification

- Overall factorized rules: PASS
- HEAD_DET orient_observed=False: 16/16
- HEAD_DET pos_source=head_det: 16/16
- FLOW_BRIDGE orient_observed=False: 0/0
- FLOW_BRIDGE pos_source=optical_flow: 0/0
- MEDIAPIPE orient_observed=True: 374/374

## Timing Additions

| Stage | Mean ms on frames where run |
|---|---:|
| Cross-agreement math | 0.002 |
| Optical-flow bridge | 0.000 |
| IMM/filter update | 0.124 |

## Honest Notes

- Back-of-head orientation is still not solved. HEAD_DET, pose, flow, and HOLD frames do not update the rotation filter as measurements.
- The real 847-frame clip did not require optical flow if `FLOW_BRIDGE frames in real run` is 0; the bridge was verified by planting a synthetic detector dropout over real frames.
- Lock-rate comparison is against v17 offline RTS output, not ground truth.

## Files

| File | Description |
|---|---|
| `pipeline_v17_fusion.py` | This pipeline |
| `live_causal_v17_fusion_stream.npz` | Factorized stream with fusion diagnostics |
| `live_causal_v17_fusion_report.json` | Benchmark and verification JSON |
| `notes_v17_fusion.md` | This report |
