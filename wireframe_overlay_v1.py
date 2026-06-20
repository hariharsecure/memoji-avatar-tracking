#!/usr/bin/env python3
"""
wireframe_overlay_v1.py — FaceMesh + canonical rig wireframe overlay.

PRIMARY (MEDIAPIPE frames — frontal/near-frontal):
  Draw the MediaPipe FaceMesh tessellation (468 canonical mesh edges mapped to the
  478 detected landmarks) overlaid directly on the subject's face. Lines deform with the subject's
  actual face geometry, expressions, and pose.  Contours (eyes / lips / oval) drawn
  in brighter colors on top.

FALLBACK (REP360 / HOLD frames — extreme profile / back-of-head):
  Render the canonical head mesh (468 verts, 898 triangles → 1365 unique edges) posed
  by yaw/pitch/roll and scaled + translated to match head_center_px + head_scale_px
  from the v15 NPZ.  This is a weak-perspective projection of the posed canonical mesh.

Outputs:
  wireframe_tracked_face_master.mp4      (full-res H.264)
  wireframe_tracked_face_preview.mp4     (<8MB, H.264)
  wireframe_tracked_face_montage.png     (5-frame grid: frontal / profile / close-up /
                                          chin-up / back-of-head)

Python: python3
"""
from __future__ import annotations
import math, os, subprocess, time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH      = "input_clip.mov"
FACE_MODEL_TASK = "models/face_landmarker.task"
CANONICAL_OBJ   = "assets/canonical_face_model.obj"
RIG_NPZ         = "./memoji_rig_stream_v13.npz"
OUT_DIR         = "."

os.makedirs(OUT_DIR, exist_ok=True)

MASTER_PATH  = f"{OUT_DIR}/wireframe_tracked_face_master.mp4"
PREVIEW_PATH = f"{OUT_DIR}/wireframe_tracked_face_preview.mp4"
MONTAGE_PATH = f"{OUT_DIR}/wireframe_tracked_face_montage.png"

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY mode — on-face tessellation
COL_TESS_MP    = (0, 200, 80)      # green-ish tessellation lines
COL_EYE        = (80, 220, 255)    # cyan eye contours
COL_LIPS       = (80, 100, 255)    # orange-red lip contour
COL_OVAL       = (200, 200, 30)    # yellow face oval

# FALLBACK mode — canonical rig wireframe
COL_FALLBACK   = (30, 180, 255)    # amber/yellow canonical rig
COL_FALLBACK_C = (255, 80, 80)     # blue accent (contour-level features)

# HUD colours
COL_HUD_MP     = (0, 255, 120)     # green HUD for MEDIAPIPE
COL_HUD_FB     = (30, 180, 255)    # amber HUD for FALLBACK

# ─────────────────────────────────────────────────────────────────────────────
# Canonical FaceMesh contour edge sets (MediaPipe spec, well-known topology)
# ─────────────────────────────────────────────────────────────────────────────
LEFT_EYE_EDGES: List[Tuple[int, int]] = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
]
RIGHT_EYE_EDGES: List[Tuple[int, int]] = [
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390),
    (390, 249), (249, 263), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
]
LIPS_OUTER_EDGES: List[Tuple[int, int]] = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (291, 409), (409, 270),
    (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40),
    (40, 185), (185, 61),
]
FACE_OVAL_EDGES: List[Tuple[int, int]] = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]

CONTOUR_EDGES_SET = frozenset(
    [tuple(sorted(e)) for e in
     LEFT_EYE_EDGES + RIGHT_EYE_EDGES + LIPS_OUTER_EDGES + FACE_OVAL_EDGES]
)

# ─────────────────────────────────────────────────────────────────────────────
# Load canonical mesh → build edge list
# ─────────────────────────────────────────────────────────────────────────────
def load_canonical_edges() -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    mesh = trimesh.load(CANONICAL_OBJ, force='mesh')
    verts = np.array(mesh.vertices, dtype=np.float64)   # (468, 3)
    faces = np.array(mesh.faces, dtype=np.int32)
    edges: set = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    return verts, faces, sorted(edges)


# ─────────────────────────────────────────────────────────────────────────────
# Mediapipe FaceLandmarker (IMAGE mode, single face)
# ─────────────────────────────────────────────────────────────────────────────
def make_face_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_TASK),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        min_tracking_confidence=0.2,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY: MediaPipe tessellation wireframe drawn directly on face
# ─────────────────────────────────────────────────────────────────────────────
def draw_mp_wireframe(canvas: np.ndarray,
                      L: np.ndarray,          # (478, 3) normalized [0,1]
                      fw: int, fh: int,
                      tess_edges: List[Tuple[int, int]]) -> np.ndarray:
    """Draw tessellation + contours from MediaPipe 478-pt face landmarks."""
    # Convert normalized to pixel coords (x=col, y=row)
    px_x = (L[:, 0] * fw).astype(np.float32)
    px_y = (L[:, 1] * fh).astype(np.float32)
    n_lm = len(L)

    # Tessellation (all 1365 edges, thin green)
    for a, b in tess_edges:
        if a >= n_lm or b >= n_lm:
            continue
        pa = (int(px_x[a]), int(px_y[a]))
        pb = (int(px_x[b]), int(px_y[b]))
        cv2.line(canvas, pa, pb, COL_TESS_MP, 1, cv2.LINE_AA)

    # Eyes — bright cyan, 1px
    for a, b in LEFT_EYE_EDGES + RIGHT_EYE_EDGES:
        if a >= n_lm or b >= n_lm:
            continue
        cv2.line(canvas,
                 (int(px_x[a]), int(px_y[a])),
                 (int(px_x[b]), int(px_y[b])),
                 COL_EYE, 1, cv2.LINE_AA)

    # Lips — brighter orange-red
    for a, b in LIPS_OUTER_EDGES:
        if a >= n_lm or b >= n_lm:
            continue
        cv2.line(canvas,
                 (int(px_x[a]), int(px_y[a])),
                 (int(px_x[b]), int(px_y[b])),
                 COL_LIPS, 1, cv2.LINE_AA)

    # Face oval — yellow, 1px
    for a, b in FACE_OVAL_EDGES:
        if a >= n_lm or b >= n_lm:
            continue
        cv2.line(canvas,
                 (int(px_x[a]), int(px_y[a])),
                 (int(px_x[b]), int(px_y[b])),
                 COL_OVAL, 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: Canonical rig posed by head-pose + placed at head_center_px/scale
# ─────────────────────────────────────────────────────────────────────────────
def draw_fallback_wireframe(canvas: np.ndarray,
                             canon_verts: np.ndarray,   # (468, 3)
                             tess_edges: List[Tuple[int, int]],
                             cx_px: float, cy_px: float, sc_px: float,
                             yaw: float, pitch: float, roll: float,
                             mode_str: str) -> np.ndarray:
    """
    Render canonical head mesh posed by yaw/pitch/roll, scaled to sc_px ear-span,
    centered at (cx_px, cy_px) via weak-perspective projection.
    """
    h, w = canvas.shape[:2]

    # Canonical ear tragion landmarks
    ear_left_3d  = canon_verts[234]
    ear_right_3d = canon_verts[454]
    ear_mid_3d   = (ear_left_3d + ear_right_3d) / 2.0
    ear_span_3d  = float(np.linalg.norm(ear_left_3d - ear_right_3d))  # ~15.33 units

    if ear_span_3d < 1e-6 or sc_px < 5.0:
        return canvas

    # Scale: map canonical ear-span → sc_px in image
    # For REP360/HOLD at profile the pose ear-span sc_px collapses.
    # Use a scale boost (calibrated from v16: bbox_w/sc ratio ~1.35 at profile)
    is_profile = abs(yaw) > 40.0
    if mode_str == 'MEDIAPIPE':
        scale_boost = 1.0
    elif is_profile:
        scale_boost = 1.35   # profile: ear-span collapses; v16 calibrated ratio (YOLO bbox_w/sc_px median=1.35 over 91 frames)
    else:
        scale_boost = 1.2    # frontal REP360: mild boost (calibration RMSE ~27px)

    scale_px_per_unit = (sc_px * scale_boost) / ear_span_3d

    # Rotate canonical mesh by yaw/pitch/roll (same Euler convention as v15)
    R = Rotation.from_euler('YXZ', [yaw, pitch, roll], degrees=True).as_matrix()

    # Center verts on ear-midpoint, then rotate, then scale
    verts_c = canon_verts - ear_mid_3d            # shift so ear-mid at origin
    verts_r = (R @ verts_c.T).T                   # rotate
    verts_s = verts_r * scale_px_per_unit         # scale to pixel units

    # Weak-perspective: x_img = cx + v_x, y_img = cy - v_y (flip y — image y is down)
    # After rotation about ear-midpoint, the projected ear-midpoint will be near (0,0) in delta
    # but changes with rotation — compute it explicitly
    ear_mid_rotated = (R @ (ear_mid_3d - ear_mid_3d))  # = (0,0,0) by construction
    x_img = cx_px + verts_s[:, 0]
    y_img = cy_px - verts_s[:, 1]

    # Z for rough depth (larger z = closer to camera in MP convention)
    # Skip backface vertices (z < mean-10) to avoid back-of-head clutter
    z_vals = verts_s[:, 2]
    z_median = float(np.median(z_vals))
    z_threshold = z_median - 5.0 * scale_px_per_unit  # generous: don't over-cull

    n_verts = len(canon_verts)
    for a, b in tess_edges:
        if a >= n_verts or b >= n_verts:
            continue
        # Back-face culling: skip edges where both endpoints are far behind
        if z_vals[a] < z_threshold and z_vals[b] < z_threshold:
            continue
        pa = (int(x_img[a]), int(y_img[a]))
        pb = (int(x_img[b]), int(y_img[b]))
        # Bounds check — don't draw wildly off-screen edges
        if (abs(pa[0]) > w * 2 or abs(pa[1]) > h * 2 or
                abs(pb[0]) > w * 2 or abs(pb[1]) > h * 2):
            continue
        cv2.line(canvas, pa, pb, COL_FALLBACK, 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(canvas: np.ndarray, fidx: int, total_f: int,
             mode: str, wire_mode: str,
             yaw: float, pitch: float, roll: float,
             anchor_src: str) -> np.ndarray:
    h, w = canvas.shape[:2]
    color = COL_HUD_MP if wire_mode == 'PRIMARY' else COL_HUD_FB
    font  = cv2.FONT_HERSHEY_SIMPLEX

    lines = [
        f"f{fidx:04d}/{total_f}  {mode}",
        f"WIRE: {wire_mode}",
        f"yaw={yaw:+.0f} pit={pitch:+.0f} rol={roll:+.0f}",
        f"src: {anchor_src[:30]}",
    ]
    y0 = 26
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(canvas, line, (10, y), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), font, 0.55, color, 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main render loop
# ─────────────────────────────────────────────────────────────────────────────
def run():
    t0 = time.time()
    print("[wireframe-v1] Loading canonical mesh and rig NPZ...")
    canon_verts, canon_faces, tess_edges = load_canonical_edges()
    print(f"  Canonical: {len(canon_verts)} verts, {len(tess_edges)} edges")

    npz = np.load(RIG_NPZ, allow_pickle=True)
    modes_arr   = npz['mode']
    yaw_arr     = npz['yaw_deg']
    pitch_arr   = npz['pitch_deg']
    roll_arr    = npz['roll_deg']
    hcx_arr     = npz['head_center_px']    # (N, 2) — ear-midpoint anchor
    hsc_arr     = npz['head_scale_px']     # (N,) — ear-span px
    anchor_arr  = npz['anchor_source']
    total_f     = len(modes_arr)
    npz_ver     = str(npz['pipeline_version'][0])
    print(f"  NPZ: {total_f} frames, pipeline_version={npz_ver}")
    # P2-B version assert: catch stale or wrong-version stream before rendering
    assert npz_ver == 'v15', (
        f"[wireframe-v1] Expected pipeline_version='v15' in {RIG_NPZ}, got '{npz_ver}'. "
        "Run pipeline_memoji_rig_v15.py to regenerate the stream before rendering."
    )

    print("[wireframe-v1] Init MediaPipe FaceLandmarker...")
    face_lmk = make_face_landmarker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {total_f} frames @ {fps:.1f}fps  {fw}x{fh}")

    # Temp output
    tmp_path = MASTER_PATH.replace('.mp4', '_tmp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (fw, fh))

    n_primary  = 0
    n_fallback = 0
    n_mp_frames_in_npz = 0
    n_mp_detect_success = 0

    # Frames we want to save for montage: frontal, profile, close-up, chin-up, back-of-head
    montage_targets = {
        50:  'frontal',
        437: 'chin-up-reacq',
        694: 'profile',
        828: 'close-up',   # P2-E fix: removed duplicate key; overwritten below by back-of-head if available
    }
    # Back-of-head: HOLD frames; first HOLD frame
    hold_frames = [i for i in range(total_f) if str(modes_arr[i]) == 'HOLD']
    if hold_frames:
        montage_targets[hold_frames[0]] = 'back-of-head'

    # Also add extreme yaw frame
    for fi in range(total_f):
        if abs(float(yaw_arr[fi])) > 80 and str(modes_arr[fi]) == 'REP360':
            montage_targets[fi] = 'extreme-profile'
            break

    saved_montage: dict = {}

    print("[wireframe-v1] Rendering frames...")
    for fidx in range(total_f):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        mode      = str(modes_arr[fidx])
        yaw       = float(yaw_arr[fidx])
        pitch     = float(pitch_arr[fidx])
        roll      = float(roll_arr[fidx])
        cx_px     = float(hcx_arr[fidx, 0])
        cy_px     = float(hcx_arr[fidx, 1])
        sc_px     = float(hsc_arr[fidx])
        anchor_src = str(anchor_arr[fidx])

        canvas = frame_bgr.copy()
        wire_mode: str

        # ── PRIMARY: try MediaPipe FaceLandmarker ──────────────────────────
        if mode == 'MEDIAPIPE':
            n_mp_frames_in_npz += 1

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        mp_result = face_lmk.detect(mp_img)

        mp_accepted = False
        if mp_result.face_landmarks:
            pts = mp_result.face_landmarks[0]
            L = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)

            # Sanity-gate: check that the detected face center is consistent
            # with the NPZ head_center_px anchor.  If the face is >300px away
            # from the NPZ anchor (ear-midpoint), the detection is a false positive
            # (e.g. f437 where MP detects the chair while the subject's head is at top-left).
            # The NPZ v15 anchor is robust (pose ear-midpoint even in extreme poses).
            det_cx = float(np.mean(L[:, 0]) * fw)
            det_cy = float(np.mean(L[:, 1] * fh))
            dist_to_anchor = math.sqrt((det_cx - cx_px) ** 2 + (det_cy - cy_px) ** 2)
            # Only accept if within 300px of the known head position
            # OR if sc_px is very small (close-up) — then anchor may not be accurate
            max_allowed_dist = max(300.0, sc_px * 2.5)
            if dist_to_anchor <= max_allowed_dist:
                canvas = draw_mp_wireframe(canvas, L, fw, fh, tess_edges)
                wire_mode = 'PRIMARY'
                n_primary += 1
                mp_accepted = True
                if mode == 'MEDIAPIPE':
                    n_mp_detect_success += 1

        if not mp_accepted:
            # ── FALLBACK: canonical rig posed from NPZ ──────────────────────
            canvas = draw_fallback_wireframe(
                canvas, canon_verts, tess_edges,
                cx_px, cy_px, sc_px,
                yaw, pitch, roll, mode
            )
            wire_mode = 'FALLBACK'
            n_fallback += 1

        # HUD
        canvas = draw_hud(canvas, fidx, total_f, mode, wire_mode,
                          yaw, pitch, roll, anchor_src)

        writer.write(canvas)

        # Save montage frames
        if fidx in montage_targets:
            label = montage_targets[fidx]
            saved_montage[label] = (canvas.copy(), fidx, wire_mode)

        if fidx % 100 == 0:
            elapsed = time.time() - t0
            fps_p = (fidx + 1) / max(elapsed, 0.01)
            print(f"  f{fidx}/{total_f}: primary={n_primary} fallback={n_fallback}  "
                  f"{fps_p:.1f}fps")

    cap.release()
    writer.release()
    face_lmk.close()

    # ── Encode with ffmpeg ─────────────────────────────────────────────────
    print("\n[wireframe-v1] Encoding master MP4 (H.264)...")
    subprocess.run([
        'ffmpeg', '-y', '-i', tmp_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', MASTER_PATH
    ], check=True, capture_output=True)
    os.remove(tmp_path)
    master_mb = os.path.getsize(MASTER_PATH) / 1e6

    print(f"[wireframe-v1] Encoding preview MP4 (<8MB)...")
    # Target 7.5MB: compute bitrate from duration
    duration_s = total_f / fps
    target_kbps = int((7.5 * 8 * 1024) / duration_s)
    subprocess.run([
        'ffmpeg', '-y', '-i', MASTER_PATH,
        '-c:v', 'libx264', '-preset', 'medium',
        '-b:v', f'{target_kbps}k', '-maxrate', f'{target_kbps*2}k',
        '-bufsize', f'{target_kbps*4}k',
        '-pix_fmt', 'yuv420p', PREVIEW_PATH
    ], check=True, capture_output=True)
    preview_mb = os.path.getsize(PREVIEW_PATH) / 1e6

    # ── Montage ────────────────────────────────────────────────────────────
    print("[wireframe-v1] Building montage...")
    montage_order = ['frontal', 'profile', 'close-up', 'chin-up-reacq',
                     'back-of-head', 'extreme-profile']
    montage_frames = []
    for label in montage_order:
        if label in saved_montage:
            frame, fidx, wmode = saved_montage[label]
            cell = frame.copy()
            cv2.putText(cell, f"{label} f{fidx} [{wmode}]", (10, fh - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(cell, f"{label} f{fidx} [{wmode}]", (10, fh - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            montage_frames.append(cell)

    if montage_frames:
        # Scale each cell to 360x640 for montage, arrange in a row
        cells = [cv2.resize(f, (360, 640)) for f in montage_frames]
        montage = np.concatenate(cells, axis=1)
        cv2.imwrite(MONTAGE_PATH, montage)
        print(f"  Montage: {len(cells)} frames → {MONTAGE_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print("WIREFRAME OVERLAY V1 — SUMMARY")
    print(f"{'='*65}")
    print(f"Total frames:          {total_f}")
    print(f"PRIMARY (MP tess):     {n_primary}  ({100*n_primary/total_f:.1f}%)")
    print(f"  of which NPZ=MP:     {n_mp_frames_in_npz}")
    print(f"  MP detect success:   {n_mp_detect_success} / {n_mp_frames_in_npz}")
    print(f"FALLBACK (rig posed):  {n_fallback}  ({100*n_fallback/total_f:.1f}%)")
    print(f"Master:   {MASTER_PATH}  ({master_mb:.1f} MB)")
    print(f"Preview:  {PREVIEW_PATH}  ({preview_mb:.1f} MB)")
    print(f"Montage:  {MONTAGE_PATH}")
    print(f"Time:     {elapsed:.0f}s")
    print(f"{'='*65}")

    return {
        'total_f':    total_f,
        'n_primary':  n_primary,
        'n_fallback': n_fallback,
        'master_mb':  master_mb,
        'preview_mb': preview_mb,
    }


if __name__ == '__main__':
    run()
