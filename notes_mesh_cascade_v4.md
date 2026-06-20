# Mesh Cascade V4 Notes

## Outputs
- Stream: `./mesh_cascade_v4_stream.npz`
- Overlay: `./mesh_cascade_v4_overlay_master.mp4`
- Preview overlay: `./mesh_cascade_v4_overlay_preview.mp4`
- Montage: `./mesh_cascade_v4_montage.png`
- Profile-fit montage: `./mesh_cascade_v4_profile_fit_montage.png`
- Interpolation montage: `./mesh_cascade_v4_interpolation_montage.png`
- JSON report: `./mesh_cascade_v4_report.json`

## Build Summary
- Single topology: MediaPipe canonical OBJ, 468 vertices / 898 faces.
- Observed tier: full-frame MediaPipe, YOLO-face crop/upscale, re-centered tight crop, centered crop, then relaxed-confidence frontal retry.
- 3DDFA tier: ONNXRuntime CPU sparse-68 fit in |yaw| < 90 with looser acceptance, but accepted only when candidate residual beats the pose baseline.
- Tail tier: bracketed temporal interpolation across all remaining unobserved spans; tagged `interpolated`, `geometry_observed=false`.
- Blendshapes: category-name remap; `_neutral` skipped; `tongueOut=0` because MediaPipe does not emit it.
- Interpolated expression: 52 blendshapes eased between bracketing observed frames; still inferred, not observed.

## Pre-Registered Tests
- Coverage: `847/847` verts populated, no NaN=True.
- Observed measured geometry: `823/847` (97.2% total); observable |yaw|<90 observed `823/828` (99.4%).
- Observed breakdown: full MP=374, zoom MP=329, tight zoom=14, centered=8, relaxed full=0, relaxed zoom=0, profile_fit=98.
- Interpolated inferred geometry: `24/847` (2.8%); true unobservable |yaw|>=90 interpolated=19, residual observable interpolated=5.
- Remaining pose_posed tail: `0/847` (0.0%).
- Low-scale observed gate: head_scale_px >= `30.0`; transition-clamped frames=25.
- Coverage by yaw bin:
  - `0_30`: total=553 observed=553 full=300 zoom=237 tight=0 centered=1 relaxed=0 profile_fit=15 interpolated=0 predicted=0
  - `30_60`: total=197 observed=192 full=61 zoom=82 tight=8 centered=4 relaxed=0 profile_fit=37 interpolated=5 predicted=0
  - `60_90`: total=78 observed=78 full=13 zoom=10 tight=6 centered=3 relaxed=0 profile_fit=46 interpolated=0 predicted=0
  - `90_180`: total=19 observed=0 full=0 zoom=0 tight=0 centered=0 relaxed=0 profile_fit=0 interpolated=19 predicted=0
- Topology: V=468 F=898 faces_sha256=`61d350723d2b69332719263ed5f04904f7baa66b814b21c76c901b621b60fd45` constant=True.
- Profile fit: attempted=103 accepted=98 rejected=5; fit_mean max=1.5715; anchor max=5.3810.
- Profile alignment vs v2 pose tail: baseline residual mean=1.0717; candidate residual mean=0.3738; better_all=True.
- No degenerate frames: bbox diag px min/median/max=90.6/178.3/542.5; running-median ratio min/max=0.721/1.778; bad_frames=0; pass=True.
- Interpolation continuity: spans=2 max mean step/head=0.1053; max p95 vertex step/head=0.1468; clamp_ref=0.1200; pass_mean=True.
- Mouth/expr Pearson r observed all observed geometry: `0.7500`.
- Mouth/expr Pearson r derivable expression frames: `0.7500` target >=0.75; derivable_n=823 profile_jaw_derived_n=98.
- Mouth/expr Pearson r frontal |yaw|<30: `0.8774` target >=0.85.
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
- Boundary pop raw before transition clamp: transitions=26 mean=0.3293 max=1.4818.
- Boundary pop clamped output: transitions=4 mean=0.0533 max=0.1053; target <=0.15, kill >0.25.
- Device: torch=mps; MediaPipe=FaceLandmarker via XNNPACK/CPU; YOLO=mps; profile_fit=3DDFA_V2 via ONNXRuntime CPU.
- Disallowed call scan pass: `True` hits={'torch_cuda_calls': [], 'blocked_render_libs': []}.
- Verify in motion: frontal run={'len': 203, 'start': 213, 'end': 415}; 3/4 run={'len': 32, 'start': 116, 'end': 147}; profile_fit run={'len': 19, 'start': 188, 'end': 206}; interpolated run={'len': 23, 'start': 430, 'end': 452}; pass observed=True pass interpolated>=8=True.

## Kill Conditions
- Zoom crop material raise >=80% observed: `True`.
- Observable observed coverage >=98%: `True`.
- Profile alignment better than pose tail: `True`.
- Corrected jawOpen r >=0.75: `True`.
- Boundary pop <=25% head scale: `True`.
- No degenerate bbox frames: `True`.
- Interpolation mean step <= clamp: `True`.
- Disallowed GPU/render calls absent: `True`.
- Kill hit: `False`.

## Interpolation Spans
- f428-428 len=1 bracket=427->429 max_mean_step=0.1053 max_p95_step=0.1468
- f430-452 len=23 bracket=429->453 max_mean_step=0.0969 max_p95_step=0.1453

## Profile Rejections
- outside_profile_yaw_gate: 19
- profile_no_yolo_face: 5

## Honest Limits
- Interpolated frames are inferred geometry and must not be treated as measured face mesh.
- True back-of-head frames in the 90-180 degree yaw bin are filled for continuity, not observed.
- Profile-fit geometry is landmark-driven canonical deformation, not a native BFM topology export.
- Profile expression is only jaw/mouth opening where the 3DDFA landmarks support it; other ARKit values are held/decayed.
- Additional zoom-observed frames inherit v17 pose/anchor metadata for the head_transform field while their vertices come from live MediaPipe landmarks remapped to full-frame coordinates.
