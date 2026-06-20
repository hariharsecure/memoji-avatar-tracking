# BFM Avatar v1 Notes

## Outputs
- 3-up master: `./subject_bfm_avatar_v1_3up_master.mp4`
- Preview: `./subject_bfm_avatar_v1_3up_preview.mp4`
- Contact sheet: `./subject_bfm_likeness_contact_sheet_v1.png`
- Identity: `./subject_bfm_identity_v1.npz`
- JSON report: `./subject_bfm_avatar_v1_report.json`

## Identity And License
- Accepted identity frames: 110
- Identity stability p90: 0.0074
- BFM asset license flag: academic-use-only (`_deps/3DDFA_V2/bfm/readme.md`); this is local proof-only, not commercial-clean.

## Render
- Alpha coverage: 100.00%
- Center error p95/head-scale: 0.1492
- BBox ratio p95: 1.1867
- Source transition pop max/head-scale: 0.1016

## Likeness Verdict
- Verdict: partial stylized resemblance; not converged on identity shape
- Better variant: stylized
- Limit: Contact sheet shows hair/beard/color cues help, but personalized BFM shape remains visually close to generic BFM. Held-out sparse reprojection and ArcFace fallback do not clear the generic-BFM baseline; photo projection has low visible-vertex fill and is comparison-only.

## Honesty
- Stylized variant uses sampled skin/feature colors plus simple vertex-color feature regions; it is not a face swap.
- Photo variant samples visible vertices from one frontal frame and falls back to sampled skin where vertices are unobserved or occluded.
- Profile frames are amber-tinted; interpolated/held frames are blue-tinted.
- The human contact sheet remains the primary gate for whether it reads as the subject.
