# Predict-Correct V8 Notes

## Scope
- v8 is a buildable-now predict/correct postpass over v7.
- Implemented pose/position confidence estimator: SO(3) error-state rotation, linear position KF, optical flow, and soft MediaPipe-Pose body prior.
- Expression/texture are not built in this iteration beyond carrying existing v7 stream fields.

## Outputs
- Stream: `./predict_correct_v8_stream.npz`
- Overlay master: `./predict_correct_v8_overlay_master.mp4`
- Overlay preview: `./predict_correct_v8_overlay_preview.mp4`
- Motion strip f425-f440: `./predict_correct_v8_motion_strip_f425_440.png`
- Reliability diagram: `./predict_correct_v8_reliability.png`
- Report: `./predict_correct_v8_report.json`
- Metrics: `./predict_correct_v8_metrics.json`

## Pre-Registered Metrics vs V7
- Pose angular jump p99: v7=24.58deg, v8=24.58deg.
- f425-f440 max pose step: v7=6.62deg, v8=6.62deg.
- f425-f440 max mean vertex jump/head: v7=0.3189, v8=0.2833.
- Reacquisition combined error: v8 median=0.1681, v7 baseline median=0.1551.
- Body prior vs v7 baseline mean combined error: body=1.2528, v7=0.1285; kill=True.
- Calibration ECE: 0.2316 over 11 scored reacquisitions.
- Confidence/status counts: {'mixed': 138, 'verified': 709}.
- Predicted rendered as verified kill: False.
- MPS/no-CUDA scan clean: True hits=[].

## Honest Floor
- Body pose is a soft prior and can be confidently wrong when the head turns independently of torso/shoulders.
- Calibration samples are limited to hidden-span reacquisitions on this clip; ECE is a probe, not a population guarantee.
- v8 pose output is confidence-labeled; only reliable face corrections produce `verified` status.
