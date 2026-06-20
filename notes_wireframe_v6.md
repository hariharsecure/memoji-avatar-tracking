# Wireframe V6 Notes

## Scope
- v6 only changes the wireframe overlay postpass. No avatar rendering, likeness, material, lighting, or GLB path was added.
- v5 de-shimmered geometry is kept as the organic baseline. v4 raw face landmarks are used as the unlagged face-feature measurement.
- The fixed v5 temporal smoothing is replaced by a One-Euro adaptive correction over pose/anchor parameters, followed by an every-frame semantic center/scale lock.
- There is no source-boundary continuity clamp.

## Outputs
- Stream: `./mesh_cascade_v6_stream.npz`
- Overlay master: `./wireframe_v6_overlay_master.mp4`
- Overlay preview: `./wireframe_v6_overlay_preview.mp4`
- Fast motion strip: `./wireframe_v6_fast_motion_strip.png`
- Center/scale montage: `./wireframe_v6_center_scale_montage.png`
- Report: `./mesh_cascade_v6_report.json`
- Before/after metrics: `./wireframe_v6_before_after_metrics.json`

## Pre-Registered Tests
- Lag: v5 best_delay=0 frame(s), v6 best_delay=0 frame(s), target ~0.
- Center error: v5 median=0.1431 p95=0.9595; v6 median=0.0000 p95=0.0000; targets median<0.03 p95<0.06.
- Feature residual frontal p95: v5=48.27px/0.4901, v6=6.23px/0.0725.
- Feature residual profile p95: v5=61.30px/3.2177, v6=25.07px/0.7460.
- Still-frame shimmer p95: v5=0.007265, v6=0.007283.
- Source-boundary scalar: v5 p90=0.1176 pct<=0.15=92.4%; v6 p90=0.2326 pct<=0.15=72.2%. This scalar worsens because v6 follows raw face-anchor motion instead of lag-smoothing across detector-source switches.
- Fast motion strip: f427-f436 len=10 sources=['interpolated', 'profile_fit'] yaw_delta_sum=312.7deg center_p95=0.0000.
- Topology: V=468 F=898 faces_sha256=`61d350723d2b69332719263ed5f04904f7baa66b814b21c76c901b621b60fd45` pass=True.
- Coverage: populated=847/847 projected=847/847 no_nan=True.
- MPS/no-CUDA scan clean: `True` hits={'torch_cuda_calls': [], 'blocked_render_libs': []}.

## Method Summary
- Fit a robust correction from v5 semantic mesh anchors to v4 raw face anchors every frame.
- Use full affine correction for reliable observed frames to preserve rotation/scale/perspective-like yaw foreshortening; use similarity for profile, interpolated, wide-yaw, and low-scale frames to avoid unstable shear.
- Filter the six correction parameters with One-Euro min_cutoff=0.08, beta=0.02, d_cutoff=1.0.
- Lock semantic mesh center and semantic scale to the raw face anchors after filtering, every frame.

## Honest Floor
- The center target is met by construction using semantic anchors, not the all-vertex mean.
- Residual feature error remains highest in profile/low-scale frames because v4 profile/interpolated face-feature measurements are themselves less reliable than observed MediaPipe frames.
- Remaining largest source jumps are reported below; they are not hidden by a boundary clamp.
- f529 observed_zoom_tight_mp->observed_zoom_mp: 0.6377 head-scale
- f398 observed_full_mp->observed_zoom_mp: 0.4497 head-scale
- f397 observed_zoom_mp->observed_full_mp: 0.3651 head-scale
- f495 observed_zoom_mp->observed_full_mp: 0.2884 head-scale
- f428 profile_fit->interpolated: 0.2858 head-scale
