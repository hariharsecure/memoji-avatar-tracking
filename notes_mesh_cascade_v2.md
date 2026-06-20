# Mesh Cascade V2 Notes

## Outputs
- Stream: `./mesh_cascade_v2_stream.npz`
- Overlay: `./mesh_cascade_v2_overlay_master.mp4`
- Preview overlay: `./mesh_cascade_v2_overlay_preview.mp4`
- Montage: `./mesh_cascade_v2_montage.png`
- JSON report: `./mesh_cascade_v2_report.json`

## Build Summary
- Single topology: MediaPipe canonical OBJ, 468 vertices / 898 faces.
- Observed tier: full-frame MediaPipe, then YOLO-face crop/upscale MediaPipe recovery.
- Tail tier: same topology posed from v17 fusion pose/anchor; tagged `pose_posed`, `mesh_conf=0`, `geometry_observed=false`.
- Blendshapes: category-name remap; `_neutral` skipped; `tongueOut=0` because MediaPipe does not emit it.
- Iteration-2 hooks left for FLAME remap and kill-gated profile 3D fit; neither is built here.

## Pre-Registered Tests
- Coverage: `847/847` verts populated, no NaN=True.
- Observed: `703/847` (83.0%). Full MP=374, zoom MP=329.
- Predicted tail: `144/847` (17.0%).
- Low-scale observed gate: head_scale_px >= `30.0`; transition-clamped frames=19.
- Coverage by yaw bin:
  - `0_30`: total=553 observed=537 predicted=16
  - `30_60`: total=197 observed=143 predicted=54
  - `60_90`: total=78 observed=23 predicted=55
  - `90_180`: total=19 observed=0 predicted=19
- Topology: V=468 F=898 faces_sha256=`61d350723d2b69332719263ed5f04904f7baa66b814b21c76c901b621b60fd45` constant=True.
- Mouth/expr Pearson r observed: `0.7545` target >=0.75.
- Mouth/expr Pearson r frontal |yaw|<30: `0.8808` target >=0.85.
- Corrected jawOpen max observed: frame `497` value `0.7302`.
- f485 check: jawOpen `0.4074`, aperture `0.1808`, source `observed_full_mp`.
- f82-91 open/close check:
  - f82: jawOpen=0.4390 aperture=0.2094 yaw=1.7 source=observed_full_mp
  - f83: jawOpen=0.3829 aperture=0.1956 yaw=1.9 source=observed_full_mp
  - f84: jawOpen=0.3675 aperture=0.2031 yaw=2.2 source=observed_full_mp
  - f85: jawOpen=0.2549 aperture=0.2041 yaw=3.6 source=observed_full_mp
  - f86: jawOpen=0.2190 aperture=0.1860 yaw=6.5 source=observed_full_mp
  - f87: jawOpen=0.0254 aperture=0.1607 yaw=13.6 source=observed_full_mp
  - f88: jawOpen=0.0181 aperture=0.1480 yaw=21.2 source=observed_full_mp
  - f89: jawOpen=0.0878 aperture=0.1155 yaw=30.1 source=observed_full_mp
  - f90: jawOpen=0.0780 aperture=0.1108 yaw=37.6 source=observed_full_mp
  - f91: jawOpen=0.0010 aperture=0.0949 yaw=47.0 source=observed_full_mp
- Boundary pop: transitions=24 mean=0.1065 max=0.1200; target <=0.15, kill >0.25.
- Device: torch=mps; MediaPipe=FaceLandmarker via XNNPACK/CPU; YOLO=mps.
- Disallowed call scan pass: `True` hits={'torch_cuda_calls': [], 'blocked_render_libs': []}.
- Verify in motion: frontal run={'len': 203, 'start': 213, 'end': 415}; 3/4 run={'len': 32, 'start': 116, 'end': 147}; pass=True.

## Kill Conditions
- Zoom crop material raise >=80% observed: `True`.
- Corrected jawOpen r >=0.70: `True`.
- Boundary pop <=25% head scale: `True`.
- Disallowed GPU/render calls absent: `True`.
- Kill hit: `False`.

## Honest Limits
- Pose-posed frames are not observed geometry and must not be treated as measured face mesh.
- The profile/back tail preserves topology and continuity, but expression is held/decayed rather than observed.
- Additional zoom-observed frames inherit v17 pose/anchor metadata for the head_transform field while their vertices come from live MediaPipe landmarks remapped to full-frame coordinates.
