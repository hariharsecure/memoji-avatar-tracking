# Mesh Avatar Driver V1 Notes

## Outputs
- Script: `./mesh_avatar_driver_v1.py`
- 3-up master MP4: `./mesh_avatar_driver_v1_master.mp4`
- Preview MP4: `./mesh_avatar_driver_v1_preview.mp4`
- Proof montage: `./mesh_avatar_driver_v1_montage.png`
- JSON report: `./mesh_avatar_driver_v1_report.json`
- Control stream: `./mesh_avatar_driver_v1_controls.npz`

## Driver Contract
- Primary driver is `mesh_cascade_v4_stream.npz` geometry: `projected_px` / `verts`.
- `arkit52_corrected` is audit-only and is not used to render the avatar.
- `head_transform` is audit-only and is not used for primary pose, placement, or scale.
- Pose comes from mesh-only PnP over canonical 468 landmarks.
- Placement and scale come from mesh landmarks: eyes, ears, top/chin, and ear span.
- Expressions come from mesh geometry ratios calibrated against high-confidence frontal observed frames.

## GLB Morph Targets
- GLB target count: `25`.
- Essential missing: `[]`.
- ARKit-52 missing from this GLB: `['cheekSquintLeft', 'cheekSquintRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'jawForward', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut']`.

## Success Metrics
- Alpha nonblank frames: `847/847` = `100.00%`.
- Avatar center error/head median `0.0297`, p95 `0.0976`.
- Avatar bbox scale ratio p95 `1.2547`.
- Source-boundary center pop max `0.0299`.
- JawOpen vs mesh aperture r frontal |yaw|<30: `0.9808`.
- JawOpen vs mesh aperture r observed non-profile: `0.9834`.
- f485 jawOpen `0.6106`, aperture `0.1797`, source `observed_full_mp`.

## Pass/Fail
- `stream_version_is_mesh_cascade_v4`: `True`
- `avatar_rendered_alpha_ge_99pct`: `True`
- `stream_mesh_coverage_100pct`: `True`
- `center_median_le_0p06`: `True`
- `center_p95_le_0p15`: `True`
- `bbox_ratio_p95_in_0p70_1p35`: `True`
- `source_boundary_pop_le_0p12`: `True`
- `jaw_r_frontal_ge_0p85`: `True`
- `jaw_r_observed_non_profile_ge_0p75`: `True`
- `f485_mouth_open_driven`: `True`
- `essential_glb_targets_present`: `True`
- `blocked_render_dependencies_absent_in_script`: `True`

## Honest Limits
- Profile-fit frames are amber. Only mouth/jaw geometry is trusted there; other controls hold/decay from prior observed frames.
- Interpolated/back-head frames are blue. Expression is held/decayed rather than invented.
- The avatar is a generic GLB head. This proof demonstrates mesh-to-rig driving, not identity likeness.
- Neutral-vs-canonical residual probe: mean `0.1701`, median `0.1264`, p95 `0.4316` normalized by ear span.
- Mouth/expression metrics are self-consistency checks between mesh-derived control and mesh aperture, not ground truth.
