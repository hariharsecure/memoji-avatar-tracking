# Face Mesh + Mouth Movement — notes_mesh_mouth.md

**Date:** 2026-06-14  
**Pipeline:** pipeline_mesh_mouth_v1.py  
**Source video:** input_clip.mov (847 frames, 720x1280, 29fps)  
**Mesh stream:** mesh_mouth_v1_stream.npz (pipeline_version='mesh_mouth_v1')  
**Rig base:** memoji_rig_stream_v17.npz (v17 fusion pipeline, head-position substrate)

---

## Purpose

Deliver the **478-pt MediaPipe FACE MESH** as a deforming tessellation that visibly
animates with the subject's mouth, jaw, expressions and blinks. The mesh IS the deliverable —
not a head-position circle or Memoji avatar. It is the substrate a future Memoji-USDZ
overlay would consume.

---

## Mode Split [MEASURED]

| Mode | Frames | % | Description |
|------|--------|---|-------------|
| MEDIAPIPE (PRIMARY) | 374 | 44.2% | Per-vertex 478-pt mesh + live 52 ARKit blendshapes |
| NO_MESH (FALLBACK) | 473 | 55.8% | Profile/back-of-head — no per-vertex mesh available |

**The 55.8% FALLBACK is honest and unavoidable:** MediaPipe FaceMesh requires a frontal
or near-frontal view. On profile/back frames (REP360 + HOLD modes from v17 pipeline),
no per-vertex geometry is available. The stream writes `NaN` for `landmarks_478` on
these frames with `mesh_quality=0.0`. The 52 blendshapes are stored with v17's decayed
values on fallback frames — they are NOT live measurements.

---

## Mouth Movement Evidence [CONFIRMED — MULTI-FRAME]

Measured exclusively on PRIMARY (MEDIAPIPE) frames with live MediaPipe detection:

| Metric | Value |
|--------|-------|
| jawOpen max (374 PRIMARY frames) | **0.5778** (f485) |
| jawOpen min (PRIMARY) | **0.0000** (f629, f665, etc.) |
| jawOpen range | **0.5778** — full open/close span |
| jawOpen std | **0.0379** |
| Consecutive PRIMARY frame pairs with jaw move > 0.02 | **14 pairs** |
| VERDICT | **CONFIRMED** |

**Multi-frame evidence (not a single frame claim):**

The following sequence was sampled directly. All frames are MEDIAPIPE mode with live
mesh + blendshapes:

| Frame | jawOpen | mouthClose | Description |
|-------|---------|------------|-------------|
| f629  | 0.000   | 0.0012     | Fully closed — mesh lips sealed |
| f082  | 0.004   | n/a        | Pre-open baseline |
| f084  | 0.057   | n/a        | Beginning to open |
| f085  | 0.058   | n/a        | Continuing open |
| f086  | 0.137   | n/a        | Opening — visible gap |
| f087  | 0.184   | n/a        | Open — inner lip ring pulled down |
| f088  | 0.074   | n/a        | Closing again |
| f090  | 0.116   | n/a        | Re-opens |
| f091  | 0.004   | n/a        | Closes |
| f251  | 0.215   | n/a        | Open with smile (mouthSmileLeft=0.197) |
| f485  | 0.578   | 0.0000     | MAX OPEN — clearest deformation |
| f488  | 0.186   | 0.0005     | Post-max, still open |
| f495  | 0.112   | 0.0004     | Closing |
| f497  | 0.007   | n/a        | Nearly closed |
| f665  | 0.000   | 0.0001     | Closed again |

**The mesh DOES deform with the subject's mouth.** The inner lip ring (yellow, landmarks 78-308-13-14)
opens/closes tracking the jawOpen value. The outer lip ring (orange, landmarks 61-291-17-0)
shows the complete mouth shape change. 14 consecutive-frame pairs exceed 0.02 jawOpen
change — this is motion in the mesh across successive frames, not a single-frame snapshot.

---

## What the Mesh Renders (PRIMARY mode)

Layer order in the video (bottom to top):

1. **Background tessellation** — 1,365 edges from canonical 468-face mesh topology,
   drawn thin gray over the face. All triangles deform with detected landmark positions.
2. **Face oval** — lime green, tracks face perimeter.
3. **Brow arches** — magenta, left + right brow landmark arcs.
4. **Eye contours** — blue, left + right eye rings.
5. **Inner lip ring** — yellow, 20 edges. This is the aperture indicator — it expands
   visibly as jawOpen rises.
6. **Outer lip ring** — orange (thickest, 3px). The main MOUTH RING — deforms clearly
   with speaking and jaw movement.
7. **Mouth landmark dots** — white dots at 28 key mouth vertices.
8. **HUD** — jawOpen bar graph (top-right), frame counter, jawOpen value.
9. **Mouth zoom inset** — 2x magnified mouth region (bottom-right).

---

## Mesh Stream Format (mesh_mouth_v1_stream.npz)

This is the clean documented stream for downstream Memoji-USDZ overlay consumption:

| Array | Shape | Type | Description |
|-------|-------|------|-------------|
| `landmarks_478` | (847, 478, 3) | float32 | Normalized coords [0,1]. NaN on NO_MESH frames. |
| `landmarks_px` | (847, 478, 2) | float32 | Pixel coords (x, y). NaN on NO_MESH frames. |
| `blendshapes_52` | (847, 52) | float32 | 52 ARKit coefficients. Live on PRIMARY; decayed on NO_MESH. |
| `arkit_names` | (52,) | str | ARKit blendshape name for each coefficient index. |
| `frame` | (847,) | int32 | Frame index 0..846. |
| `mode` | (847,) | str | 'MEDIAPIPE' or 'NO_MESH'. |
| `jawOpen` | (847,) | float32 | Shortcut: blendshapes_52[:, 24]. |
| `mouthClose` | (847,) | float32 | Shortcut: blendshapes_52[:, 26]. |
| `mesh_quality` | (847,) | float32 | 1.0=full live mesh; 0.0=no mesh. |
| `yaw_deg` | (847,) | float32 | Head yaw from v17 rig. |
| `pitch_deg` | (847,) | float32 | Head pitch from v17 rig. |
| `roll_deg` | (847,) | float32 | Head roll from v17 rig. |
| `pipeline_version` | (1,) | str | 'mesh_mouth_v1' |

**How a Memoji-USDZ overlay would consume this:**
- Use `mesh_quality > 0.5` to select PRIMARY frames (478 live vertices available).
- `landmarks_478[fidx]` → drive the USDZ face mesh vertex positions directly.
- `blendshapes_52[fidx]` → drive the 52 ARKit blend shape targets on the Memoji rig.
- On NO_MESH frames (`mesh_quality == 0.0`): use held/decayed blendshapes + head
  position from yaw/pitch/roll for a pose-only render. Per-vertex geometry unavailable.

**Key ARKit indices for expression control:**

| Index | Name | Primary use |
|-------|------|-------------|
| 24 | jawOpen | Mouth aperture (0=closed, 1=fully open) |
| 26 | mouthClose | Lip press |
| 43 | mouthSmileLeft | Smile |
| 44 | mouthSmileRight | Smile |
| 8  | eyeBlinkLeft | Blink |
| 9  | eyeBlinkRight | Blink |
| 2  | browInnerUp | Brow raise |
| 0  | browDownLeft | Brow furrow |

---

## Honest Limits

1. **44.2% PRIMARY coverage** — the 55.8% fallback (REP360 + HOLD) has no per-vertex
   mesh. This is a fundamental MediaPipe constraint: FaceMesh requires a frontal view.
   The subject's video has significant profile/walking content. The fallback frames have
   `landmarks_478 = NaN` and `mesh_quality = 0.0`.

2. **jawOpen range 0–0.578** — The maximum measured jawOpen is 0.578 (f485). A fully
   open yawn would be ~0.8–1.0. The range observed covers normal speaking/expression.
   The mesh deformation IS visible and tracks real jaw motion.

3. **No 3D vertex depth for USDZ** — `landmarks_478[:, :, 2]` is the MediaPipe
   pseudo-depth (negative z = toward camera). It is not calibrated camera-space depth.
   For a true USDZ overlay, camera intrinsics and a fitted 3D face model would be
   needed to convert to world coordinates. The 2D pixel landmarks are immediately
   usable for 2D overlay.

4. **Blendshape quality on fallback** — The 52 ARKit values on NO_MESH frames are
   carried from v17's BSHP_DECAY=0.92 decay toward neutral. They are NOT live
   measurements. Only MEDIAPIPE frames have real-time blendshapes.

---

## Outputs

| File | Size | Description |
|------|------|-------------|
| `mesh_mouth_v1_master.mp4` | 15.3 MB | Full-res H.264 mesh render |
| `mesh_mouth_v1_preview.mp4` | 7.7 MB | web-preview-safe <8MB |
| `mesh_mouth_v1_montage.png` | — | 5-panel: closed/opening/max-open/smile/neutral |
| `mesh_mouth_v1_stream.npz` | 3.3 MB | Per-frame 478 landmarks + 52 blendshapes |
| `mesh_mouth_v1_report.json` | — | Per-frame stats + evidence |
| `mesh_proof_*.jpg` | — | 8 individual proof frames (mouth progression) |
| `pipeline_mesh_mouth_v1.py` | — | Producer script |
| `notes_mesh_mouth.md` | — | This report |

---

## Verdict

**The mesh deforms with the subject's mouth and expressions — confirmed across 14 consecutive
frame pairs with measured jaw movement > 0.02, and a full jawOpen range of 0–0.578
across 374 PRIMARY frames.**

The tessellation, inner lip ring, and outer lip ring visibly change shape frame-to-frame
as the subject's jaw opens and closes. This is the Memoji substrate: 478 live vertex positions +
52 ARKit blendshape coefficients per frontal frame, in a clean documented NPZ format.

The 55.8% fallback (profile/back-of-head) is honest — MediaPipe FaceMesh is a frontal
tracker. On those frames, `mesh_quality=0.0` and `landmarks_478=NaN`. Per-vertex face
mesh on extreme profile frames would require a separate 3D fitting step (e.g., FLAME,
EMOCA) — out of scope for this pass.
