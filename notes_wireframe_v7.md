# Wireframe V7 Notes

## Scope
- v7 builds on v6 and keeps v6 MediaPipe-observed geometry unchanged.
- The v7 change is limited to profile/interpolated averted frames: quaternion constant-angular-velocity pose replacement, outlier rejection, optical-flow head-motion validation, and a rebuilt mesh pose arc.
- v6 semantic center/scale lock is preserved on observed frames; rebuilt averted frames use the v6 target center unless it contradicts the head optical-flow track, where a weighted flow-center override is used.

## Outputs
- Stream: `./mesh_cascade_v7_stream.npz`
- Overlay master: `./wireframe_v7_overlay_master.mp4`
- Overlay preview: `./wireframe_v7_overlay_preview.mp4`
- Pose-flip motion strip f425-f440: `./wireframe_v7_pose_flip_motion_strip_f425_440.png`
- Center/scale montage: `./wireframe_v7_center_scale_montage.png`
- Report: `./mesh_cascade_v7_report.json`
- Before/after metrics: `./wireframe_v7_before_after_pose_metrics.json`

## Pre-Registered Tests
- Pose angular step p99: v6=44.53deg, v7=24.58deg.
- f425-f440 max pose step: v6=138.89deg, v7=6.62deg.
- f429 flip: raw yaw=-1.55deg; v7 unwrapped yaw=-127.53deg.
- f425-f440 max mean vertex jump/head: v6=0.4366, v7=0.3189.
- Lag: v6 best_delay=0 frame(s), v7 best_delay=0 frame(s), target ~0.
- Center error: v6 median=0.0000 p95=0.0000; v7 median=0.0000 p95=0.0000.
- Observed frame preservation: max projected delta vs v6 observed frames=0.000000px; flow-center override frames=37.
- Feature residual profile p95: v6=25.07px/0.7460, v7=45.60px/1.2265.
- Motion strip: f425-f440 len=16 sources=['interpolated', 'profile_fit'] yaw_delta_sum=95.1deg center_p95=0.1494.
- Topology: V=468 F=898 faces_sha256=`61d350723d2b69332719263ed5f04904f7baa66b814b21c76c901b621b60fd45` pass=True.
- Coverage: populated=847/847 projected=847/847 no_nan=True.
- MPS/no-CUDA scan clean: `True` hits={'torch_cuda_calls': [], 'blocked_render_libs': []}.

## Method Summary
- Convert `head_transform` rotations to unit quaternions with polar/SVD normalization.
- For each profile/interpolated span, estimate entry angular velocity from the preceding reliable observed frames and propagate that rotation arc through the averted span.
- Reject raw pose reads when they diverge from the CAV prediction, produce a neighbor spike, or contradict the optical-flow head-motion check.
- Rebuild profile/interpolated mesh frames from stabilized neutral pose plus smoothstep-interpolated observed residuals, then relock semantic center/scale to a weighted v6/flow center target.

## Honest Floor
- v7 removes the f429 pose flip from the emitted pose stream; residual profile feature error can remain because the back/profile frames still lack true face observations.
- Profile feature residual p95 rises versus v6 because v7 prioritizes head-pose continuity and flow-visible head anchoring over matching unreliable averted face-feature anchors.
- Optical flow is used as a validation signal, not as a full segmentation tracker; low-texture or motion-blurred head regions can lower flow confidence.
- Motion cross-verification against an external reference is pending; this run produced the motion strip and metrics for that check.
- Remaining largest source jumps are reported below; no source-boundary clamp is hidden.
- f529 observed_zoom_tight_mp->observed_zoom_mp: 0.6377 head-scale
- f398 observed_full_mp->observed_zoom_mp: 0.4497 head-scale
- f397 observed_zoom_mp->observed_full_mp: 0.3651 head-scale
- f428 profile_fit->interpolated: 0.3189 head-scale
- f495 observed_zoom_mp->observed_full_mp: 0.2884 head-scale
