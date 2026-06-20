# Wireframe Perfect V5 Notes

## Scope
- Only wireframe/mesh quality was changed. No avatar rendering, likeness, material, lighting, or GLB path was added.
- v5 is a source-neutral postpass over `mesh_cascade_v4_stream.npz`: every source is decomposed into the same canonical 2D pose basis, pose is smoothed globally, and local shape is smoothed motion-compensated.
- There is no transition-specific continuity clamp in v5.

## Outputs
- Stream: `./mesh_cascade_v5_stream.npz`
- Overlay master: `./wireframe_perfect_v5_overlay_master.mp4`
- Overlay preview: `./wireframe_perfect_v5_overlay_preview.mp4`
- Montage: `./wireframe_perfect_v5_montage.png`
- Report: `./mesh_cascade_v5_report.json`
- Before/after metrics: `./wireframe_perfect_before_after_metrics.json`

## Registered Tests
- V4 raw pre-clamp seam baseline: transitions=26 mean=0.3293 max=1.4818.
- V4 saved source transitions: transitions=79 mean=0.1314 p90=0.2245 max=0.6581 pct<=0.15=77.2%.
- V5 emitted raw source transitions, no clamp: transitions=79 mean=0.0628 p90=0.1176 max=0.3350 pct<=0.15=92.4% pass_90pct=True.
- Profile shape-only jitter p95: before=0.1447, after=0.0232.
- Frontal shape-only jitter p95: before=0.0235, after=0.0133; after profile/frontal ratio=1.74.
- Topology: V=468 F=898 faces_sha256=`61d350723d2b69332719263ed5f04904f7baa66b814b21c76c901b621b60fd45` pass=True.
- Coverage: populated=847/847 projected=847/847 no_nan=True.
- MPS/no-CUDA scan clean: `True` hits={'torch_cuda_calls': [], 'blocked_render_libs': []}.
- Verify-in-motion span f397-f430 len=34 sources=['interpolated', 'observed_centered_mp', 'observed_full_mp', 'observed_zoom_mp', 'profile_fit'] max_step=0.3162 p95_step=0.2917; visual strip frames=[424, 425, 426, 427, 428, 429, 430, 431].

## Profile Interior
- The 3DDFA profile-fit silhouette is kept as a useful measurement, but its eyes/mouth/interior are not treated as fully trustworthy.
- v5 down-weights profile-fit eye/mouth/interior vertices and lets bracketed MediaPipe observations dominate those vertices through the profile spans.
- Bracketed profile spans: 13/13.

## Honest Floor
- The 90% seam target is met without the v4 transition clamp.
- The remaining max seam floor is not hidden. It is concentrated in low-scale/reacquisition regions where neighboring sources disagree about the face extent and anchor.
- f397 observed_zoom_mp->observed_full_mp: 0.3350 head-scale
- f428 profile_fit->interpolated: 0.3162 head-scale
- f429 interpolated->profile_fit: 0.2982 head-scale
- f398 observed_full_mp->observed_zoom_mp: 0.2873 head-scale
- f794 profile_fit->observed_zoom_tight_mp: 0.1792 head-scale

## Method Summary
- Fit a robust reflected 2D similarity from canonical anchors to every frame, using the same anchor set for full MP, zoom MP, tight/centered MP, profile_fit, and interpolated frames.
- Smooth that pose track with a centered robust temporal filter, not a source-boundary branch.
- Convert vertices into the shared canonical-local basis, smooth local shape with source and vertex trust, and project back through the unified pose track.
- Smooth profile fits more strongly and replace untrusted profile interior with bracketed MediaPipe-local shape while retaining profile silhouette contribution.
