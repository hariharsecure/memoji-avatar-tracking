# Mesh Cascade V3 Notes

## Outputs
- Stream: `./mesh_cascade_v3_stream.npz`
- Overlay: `./mesh_cascade_v3_overlay_master.mp4`
- Preview overlay: `./mesh_cascade_v3_overlay_preview.mp4`
- Montage: `./mesh_cascade_v3_montage.png`
- Profile-fit montage: `./mesh_cascade_v3_profile_fit_montage.png`
- JSON report: `./mesh_cascade_v3_report.json`

## Build Summary
- Single topology: MediaPipe canonical OBJ, 468 vertices / 898 faces.
- Observed tier: full-frame MediaPipe, then YOLO-face crop/upscale MediaPipe recovery.
- Profile tier: 3DDFA_V2 ONNXRuntime CPU sparse-68 profile fit, accepted only in |yaw| 30-90 and remapped onto the same canonical 468 vertices.
- Tail tier: same topology posed from v17 fusion pose/anchor; tagged `pose_posed`, `mesh_conf=0`, `geometry_observed=false`.
- Blendshapes: category-name remap; `_neutral` skipped; `tongueOut=0` because MediaPipe does not emit it.
- Profile expression: only jawOpen is derived from 3DDFA mouth landmarks when available; all other ARKit slots hold/decay from the previous observed stream.

## Pre-Registered Tests
- Coverage: `847/847` verts populated, no NaN=True.
- Observed: `788/847` (93.0%). Full MP=374, zoom MP=329, profile_fit=85.
- Predicted tail: `59/847` (7.0%).
- Low-scale observed gate: head_scale_px >= `30.0`; transition-clamped frames=32.
- Coverage by yaw bin:
  - `0_30`: total=553 observed=537 full=300 zoom=237 profile_fit=0 predicted=16
  - `30_60`: total=197 observed=185 full=61 zoom=82 profile_fit=42 predicted=12
  - `60_90`: total=78 observed=66 full=13 zoom=10 profile_fit=43 predicted=12
  - `90_180`: total=19 observed=0 full=0 zoom=0 profile_fit=0 predicted=19
- Topology: V=468 F=898 faces_sha256=`61d350723d2b69332719263ed5f04904f7baa66b814b21c76c901b621b60fd45` constant=True.
- Profile fit: attempted=103 accepted=85 rejected=18; fit_mean max=0.6496; anchor max=2.5418.
- Profile alignment vs v2 pose tail: baseline residual mean=0.5814; candidate residual mean=0.2530; better_all=True.
- Mouth/expr Pearson r observed all observed geometry: `0.7509`.
- Mouth/expr Pearson r derivable expression frames: `0.7509` target >=0.70; derivable_n=788 profile_jaw_derived_n=85.
- Mouth/expr Pearson r frontal |yaw|<30: `0.8811` target >=0.85.
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
- Boundary pop raw before transition clamp: transitions=34 mean=0.3198 max=1.2907.
- Boundary pop clamped output: transitions=18 mean=0.1116 max=0.1200; target <=0.15, kill >0.25.
- Device: torch=mps; MediaPipe=FaceLandmarker via XNNPACK/CPU; YOLO=mps; profile_fit=3DDFA_V2 via ONNXRuntime CPU.
- Disallowed call scan pass: `True` hits={'torch_cuda_calls': [], 'blocked_render_libs': []}.
- Verify in motion: frontal run={'len': 203, 'start': 213, 'end': 415}; 3/4 run={'len': 32, 'start': 116, 'end': 147}; profile_fit run={'len': 29, 'start': 178, 'end': 206}; pass=True.

## Kill Conditions
- Zoom crop material raise >=80% observed: `True`.
- Profile fit material raise >=93% observed: `True`.
- Profile alignment better than v2 pose tail: `True`.
- Corrected jawOpen r >=0.70: `True`.
- Boundary pop <=25% head scale: `True`.
- Disallowed GPU/render calls absent: `True`.
- Kill hit: `False`.

## Profile Rejections
- outside_profile_yaw_gate: 35
- profile_head_scale_too_small: 6
- profile_no_yolo_face: 6
- profile_anchor_gate_2.851: 1
- profile_anchor_gate_3.185: 1
- profile_anchor_gate_3.225: 1
- profile_anchor_gate_3.783: 1
- profile_fit_mean_gate_0.897: 1
- profile_fit_mean_gate_1.065: 1
- profile_fit_mean_gate_1.270: 1
- profile_fit_mean_gate_1.323: 1
- profile_fit_mean_gate_1.401: 1
- profile_fit_mean_gate_1.571: 1
- profile_not_better_than_pose_0.154_vs_0.145: 1
- profile_not_better_than_pose_0.220_vs_0.209: 1

## Honest Limits
- Pose-posed frames are not observed geometry and must not be treated as measured face mesh.
- True back-of-head frames in the 90-180 degree yaw bin remain pose_posed; this face-fit lane does not recover them.
- Profile-fit geometry is landmark-driven canonical deformation, not a native BFM topology export.
- Profile expression is only jaw/mouth opening where the 3DDFA landmarks support it; other ARKit values are held/decayed.
- Additional zoom-observed frames inherit v17 pose/anchor metadata for the head_transform field while their vertices come from live MediaPipe landmarks remapped to full-frame coordinates.
