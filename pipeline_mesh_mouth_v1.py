#!/usr/bin/env python3
"""
pipeline_mesh_mouth_v1.py — Face MESH + MOUTH MOVEMENT delivery.

PURPOSE
-------
the subject's goal: the 478-pt MediaPipe FACE MESH deforming WITH the subject's mouth and
expressions — the substrate for a future Memoji/USDZ overlay. Not a head
position tracker — the MESH itself, visibly animated.

DELIVERABLES
------------
1. mesh_mouth_v1_master.mp4      — full-res mesh render: 478-pt tessellation
                                   on face with color-coded mouth/jaw/eye/brow
                                   contours + blendshape HUD (jawOpen bar graph).
   mesh_mouth_v1_preview.mp4     — a smaller <8MB preview version.

2. mesh_mouth_v1_montage.png     — 5-panel montage:
                                   [mouth-closed | mouth-opening | mouth-open MAX
                                    | smile+brow | blink-close]
                                   with side-by-side mesh vs reference.

3. mesh_mouth_v1_stream.npz      — per-frame MESH STREAM:
                                   - landmarks_478: (N, 478, 3) normalized coords
                                   - landmarks_px:  (N, 478, 2) pixel coords
                                   - blendshapes_52: (N, 52) ARKit coefficients
                                   - arkit_names: (52,)
                                   - frame: (N,) frame index
                                   - mode: (N,) 'MEDIAPIPE' or 'NO_MESH'
                                   - jawOpen: (N,) shortcut accessor
                                   - mouthClose: (N,) shortcut accessor
                                   - mesh_quality: (N,) [0.0=no_mesh, 1.0=full_mesh]
                                   - pipeline_version: 'mesh_mouth_v1'

4. mesh_mouth_v1_report.json     — frame-by-frame stats, mouth-movement evidence.

DESIGN
------
PRIMARY (MEDIAPIPE mode, ~44% of frames):
  Re-run MediaPipe FaceLandmarker WITH output_face_blendshapes=True.
  Draw full 478-pt tessellation on face.
  Color scheme:
    - Thin white:  all tessellation edges (background mesh)
    - Bright cyan: inner lip (upper + lower lip detail)
    - Hot orange:  outer lip / jaw contour — MOUTH RING
    - Lime green:  face oval
    - Electric blue: eye contours
    - Magenta:     brow arch
  BLENDSHAPE HUD: jawOpen, mouthSmileLeft, eyeBlinkLeft — bar graph top-right.
  Mouth region ZOOM (bottom-right inset): 2x magnified mouth landmark area.

FALLBACK (REP360 / HOLD / NO_MESH — ~56% of frames):
  NO per-vertex mesh available for non-frontal/back frames.
  Show label "NO MESH (fallback — profile/back)" with yaw/pitch.
  Write NaN for landmarks_478 on these frames (honest).
  The blendshapes from v17 NPZ (decay-interpolated) are written, with
  mesh_quality=0.0 to signal downstream that geometry is unavailable.

Python: python3
"""
from __future__ import annotations
import json, math, os, subprocess, time
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH      = "input_clip.mov"
FACE_MODEL_TASK = "models/face_landmarker.task"
RIG_NPZ         = "./memoji_rig_stream_v17.npz"
OUT_DIR         = "."

MASTER_PATH    = f"{OUT_DIR}/mesh_mouth_v1_master.mp4"
PREVIEW_PATH   = f"{OUT_DIR}/mesh_mouth_v1_preview.mp4"
MONTAGE_PATH   = f"{OUT_DIR}/mesh_mouth_v1_montage.png"
STREAM_PATH    = f"{OUT_DIR}/mesh_mouth_v1_stream.npz"
REPORT_PATH    = f"{OUT_DIR}/mesh_mouth_v1_report.json"

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe tessellation connection list (canonical 468→478 face mesh topology)
# Use mp.solutions.face_mesh's FACEMESH_TESSELATION
# ─────────────────────────────────────────────────────────────────────────────
# These are the canonical 468-triangle tessellation edges from MediaPipe
# face_mesh.FACEMESH_TESSELATION — we reconstruct them without importing solutions
# by using the face_landmarker result directly with the tessellation connections
# known from MediaPipe source (frozenset of (int, int) tuples, 1365 edges)
#
# For 478-pt model: indices 468-477 are extra iris landmarks (not in canonical
# tessellation). The mesh topology for 0-467 is identical.

# Full 1365-edge canonical tessellation (from mp.solutions.face_mesh.FACEMESH_TESSELATION)
# We load via a known reliable method: read from trimesh canonical OBJ
def build_tess_edges_from_mp() -> List[Tuple[int,int]]:
    """Build tessellation edge list from MediaPipe solutions FACEMESH_TESSELATION."""
    try:
        import mediapipe.python.solutions.face_mesh as fm
        edges = set()
        for a, b in fm.FACEMESH_TESSELATION:
            edges.add((min(a,b), max(a,b)))
        return sorted(edges)
    except Exception:
        # Fallback: load from OBJ
        import trimesh
        CANONICAL_OBJ = "assets/canonical_face_model.obj"
        mesh = trimesh.load(CANONICAL_OBJ, force='mesh')
        faces = np.array(mesh.faces, dtype=np.int32)
        edges = set()
        for f in faces:
            for i in range(3):
                a, b = int(f[i]), int(f[(i+1)%3])
                edges.add((min(a,b), max(a,b)))
        return sorted(edges)


# ─────────────────────────────────────────────────────────────────────────────
# Face mesh contour landmark indices (MediaPipe canonical)
# ─────────────────────────────────────────────────────────────────────────────

# Outer lip ring — the MOUTH contour (most important for the subject's request)
LIPS_OUTER: List[Tuple[int,int]] = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (291, 409), (409, 270),
    (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40),
    (40, 185), (185, 61),
]
# Inner lip (reveals how open the mouth is)
LIPS_INNER: List[Tuple[int,int]] = [
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308), (308, 415), (415, 310),
    (310, 311), (311, 312), (312, 13), (13, 82), (82, 81), (81, 80),
    (80, 191), (191, 78),
]
# Eye contours
LEFT_EYE: List[Tuple[int,int]] = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
]
RIGHT_EYE: List[Tuple[int,int]] = [
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390),
    (390, 249), (249, 263), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
]
# Brow arches
LEFT_BROW: List[Tuple[int,int]] = [
    (46, 53), (53, 52), (52, 65), (65, 55), (55, 70), (70, 63), (63, 105),
    (105, 66), (66, 107), (107, 46),
]
RIGHT_BROW: List[Tuple[int,int]] = [
    (276, 283), (283, 282), (282, 295), (295, 285), (285, 300), (300, 293),
    (293, 334), (334, 296), (296, 336), (336, 276),
]
# Face oval
FACE_OVAL: List[Tuple[int,int]] = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]

# Key mouth landmarks for zoom region (lip corners + upper/lower lip center)
# 0=upper lip center, 17=lower lip center, 61=left lip corner, 291=right lip corner
# 13=inner upper lip, 14=inner lower lip
MOUTH_KEY_LMK = [0, 13, 14, 17, 61, 78, 87, 88, 91, 146, 181,
                 267, 269, 270, 291, 308, 311, 312, 314, 317, 318,
                 321, 324, 375, 402, 405, 409, 415]

# ─────────────────────────────────────────────────────────────────────────────
# Colors (BGR)
# ─────────────────────────────────────────────────────────────────────────────
COL_TESS      = (180, 180, 180)   # light gray — full tessellation background
COL_OVAL      = (50, 220, 50)     # lime green — face oval
COL_LIPS_OUT  = (0, 100, 255)     # hot orange — outer lip (MOUTH RING — most visible)
COL_LIPS_IN   = (50, 200, 255)    # yellow — inner lip (openness indicator)
COL_EYE       = (255, 200, 50)    # electric blue — eyes
COL_BROW      = (255, 60, 200)    # magenta — brows
COL_HUD_MP    = (0, 255, 120)     # green HUD for MEDIAPIPE mode
COL_HUD_FB    = (40, 140, 200)    # amber for FALLBACK
COL_BAR_JAW   = (0, 100, 255)     # orange — jawOpen bar
COL_BAR_SMILE = (0, 220, 100)     # green — smile bar
COL_BAR_BLINK = (200, 50, 255)    # pink — blink bar
COL_BAR_BROW  = (255, 100, 0)     # cyan — browInnerUp bar

# Landmark point sizes
TESS_LW       = 1   # tessellation line width
CONTOUR_LW    = 2   # contour line width
MOUTH_LW      = 2   # mouth ring line width


def make_face_landmarker_with_blendshapes():
    """Create FaceLandmarker with blendshapes + face transform enabled."""
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_TASK),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        min_tracking_confidence=0.2,
        output_face_blendshapes=True,                   # KEY: get 52 ARKit blendshapes
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def draw_mesh_primary(canvas: np.ndarray,
                      L: np.ndarray,               # (478, 3) normalized
                      fw: int, fh: int,
                      tess_edges: List[Tuple[int,int]],
                      blendshapes_52: np.ndarray,
                      arkit_names: List[str]) -> np.ndarray:
    """
    Draw the full 478-pt face mesh on canvas.
    Layer order: tessellation (background) → oval → brows → eyes → inner lip → outer lip
    """
    px_x = (L[:, 0] * fw).astype(np.float32)
    px_y = (L[:, 1] * fh).astype(np.float32)
    n_lm = len(L)

    # 1. Background tessellation — all edges, thin gray
    for a, b in tess_edges:
        if a >= n_lm or b >= n_lm:
            continue
        pa = (int(px_x[a]), int(px_y[a]))
        pb = (int(px_x[b]), int(px_y[b]))
        cv2.line(canvas, pa, pb, COL_TESS, TESS_LW, cv2.LINE_AA)

    # 2. Face oval — lime green
    for a, b in FACE_OVAL:
        if a >= n_lm or b >= n_lm: continue
        cv2.line(canvas, (int(px_x[a]), int(px_y[a])),
                 (int(px_x[b]), int(px_y[b])), COL_OVAL, CONTOUR_LW, cv2.LINE_AA)

    # 3. Brows — magenta
    for a, b in LEFT_BROW + RIGHT_BROW:
        if a >= n_lm or b >= n_lm: continue
        cv2.line(canvas, (int(px_x[a]), int(px_y[a])),
                 (int(px_x[b]), int(px_y[b])), COL_BROW, CONTOUR_LW, cv2.LINE_AA)

    # 4. Eyes — blue
    for a, b in LEFT_EYE + RIGHT_EYE:
        if a >= n_lm or b >= n_lm: continue
        cv2.line(canvas, (int(px_x[a]), int(px_y[a])),
                 (int(px_x[b]), int(px_y[b])), COL_EYE, CONTOUR_LW, cv2.LINE_AA)

    # 5. Inner lip — yellow (shows mouth opening)
    for a, b in LIPS_INNER:
        if a >= n_lm or b >= n_lm: continue
        cv2.line(canvas, (int(px_x[a]), int(px_y[a])),
                 (int(px_x[b]), int(px_y[b])), COL_LIPS_IN, MOUTH_LW, cv2.LINE_AA)

    # 6. Outer lip — BRIGHT ORANGE ring (most important visible indicator)
    for a, b in LIPS_OUTER:
        if a >= n_lm or b >= n_lm: continue
        cv2.line(canvas, (int(px_x[a]), int(px_y[a])),
                 (int(px_x[b]), int(px_y[b])), COL_LIPS_OUT, MOUTH_LW + 1, cv2.LINE_AA)

    # 7. Landmark dots at mouth key points (visible deformation dots)
    for lm_idx in MOUTH_KEY_LMK:
        if lm_idx >= n_lm: continue
        cv2.circle(canvas, (int(px_x[lm_idx]), int(px_y[lm_idx])),
                   2, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas


def draw_blendshape_hud(canvas: np.ndarray,
                         blendshapes_52: np.ndarray,
                         arkit_names: List[str],
                         jaw_open_idx: int, smile_l_idx: int,
                         blink_l_idx: int, brow_idx: int,
                         fw: int, fh: int) -> np.ndarray:
    """
    Draw a compact blendshape bar-graph in top-right corner.
    4 key blendshapes: jawOpen / mouthSmileLeft / eyeBlinkLeft / browInnerUp
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar_w = 120       # max bar width in px
    bar_h = 14
    x0 = fw - bar_w - 60   # right-align with label
    y0 = 10
    pad = 4

    entries = [
        ('jawOpen',        blendshapes_52[jaw_open_idx],  COL_BAR_JAW),
        ('smileL',         blendshapes_52[smile_l_idx],   COL_BAR_SMILE),
        ('blinkL',         blendshapes_52[blink_l_idx],   COL_BAR_BLINK),
        ('browUp',         blendshapes_52[brow_idx],      COL_BAR_BROW),
    ]

    for i, (label, val, col) in enumerate(entries):
        y = y0 + i * (bar_h + pad)
        # Background bar
        cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + bar_h), (30, 30, 30), -1)
        # Value bar
        filled = int(val * bar_w)
        if filled > 0:
            cv2.rectangle(canvas, (x0, y), (x0 + filled, y + bar_h), col, -1)
        # Label
        lbl = f"{label}: {val:.3f}"
        cv2.putText(canvas, lbl, (x0 - 8, y + bar_h - 2), font, 0.38, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (x0 - 8, y + bar_h - 2), font, 0.38, (255,255,255), 1, cv2.LINE_AA)

    return canvas


def draw_mouth_zoom_inset(canvas: np.ndarray,
                           L: np.ndarray, fw: int, fh: int,
                           tess_edges: List[Tuple[int,int]]) -> np.ndarray:
    """
    Draw a 2× magnified mouth-region inset in bottom-right corner.
    Region: landmarks 0, 17, 61, 291 define the mouth bounding box.
    """
    n_lm = len(L)
    px_x = (L[:, 0] * fw).astype(np.float32)
    px_y = (L[:, 1] * fh).astype(np.float32)

    # Mouth region bounds
    mouth_pts = [p for p in [0, 17, 61, 291, 13, 14, 78, 308,
                              84, 314, 87, 317, 88, 318, 91, 321, 146, 375]
                 if p < n_lm]
    if not mouth_pts:
        return canvas

    xs = [px_x[p] for p in mouth_pts]
    ys = [px_y[p] for p in mouth_pts]
    mx0 = max(0, int(min(xs)) - 30)
    my0 = max(0, int(min(ys)) - 20)
    mx1 = min(fw, int(max(xs)) + 30)
    my1 = min(fh, int(max(ys)) + 20)

    if mx1 <= mx0 or my1 <= my0:
        return canvas

    # Crop mouth region
    mouth_crop = canvas[my0:my1, mx0:mx1].copy()
    zoom_h = 120
    zoom_w = int(zoom_h * (mx1 - mx0) / (my1 - my0))
    if zoom_w < 10: return canvas
    zoom = cv2.resize(mouth_crop, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)

    # Draw mouth landmarks on zoom
    scale_x = zoom_w / (mx1 - mx0)
    scale_y = zoom_h / (my1 - my0)
    for a, b in LIPS_OUTER + LIPS_INNER:
        if a >= n_lm or b >= n_lm: continue
        zpa = (int((px_x[a] - mx0) * scale_x), int((px_y[a] - my0) * scale_y))
        zpb = (int((px_x[b] - mx0) * scale_x), int((px_y[b] - my0) * scale_y))
        color = COL_LIPS_OUT if (a, b) in set(map(tuple, LIPS_OUTER)) else COL_LIPS_IN
        if (0 <= zpa[0] < zoom_w and 0 <= zpa[1] < zoom_h and
                0 <= zpb[0] < zoom_w and 0 <= zpb[1] < zoom_h):
            cv2.line(zoom, zpa, zpb, color, 2, cv2.LINE_AA)

    # Border
    cv2.rectangle(zoom, (0, 0), (zoom_w-1, zoom_h-1), (0, 180, 255), 2)

    # Place in bottom-right of canvas
    h, w = canvas.shape[:2]
    x_off = w - zoom_w - 8
    y_off = h - zoom_h - 8
    if x_off > 0 and y_off > 0:
        canvas[y_off:y_off+zoom_h, x_off:x_off+zoom_w] = zoom

    # Label
    cv2.putText(canvas, "MOUTH x2", (x_off, y_off - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1, cv2.LINE_AA)

    return canvas


def draw_hud_primary(canvas: np.ndarray, fidx: int, total_f: int,
                     yaw: float, pitch: float, roll: float,
                     jaw_open: float, mode_str: str) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    col = COL_HUD_MP
    lines = [
        f"f{fidx:04d}/{total_f}  MESH:PRIMARY",
        f"yaw={yaw:+.0f} pit={pitch:+.0f} rol={roll:+.0f}",
        f"jawOpen={jaw_open:.3f}",
    ]
    for i, line in enumerate(lines):
        y = 26 + i * 22
        cv2.putText(canvas, line, (10, y), font, 0.52, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), font, 0.52, col, 1, cv2.LINE_AA)
    return canvas


def draw_hud_fallback(canvas: np.ndarray, fidx: int, total_f: int,
                      yaw: float, pitch: float, roll: float,
                      mode_str: str) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    col = COL_HUD_FB
    h, w = canvas.shape[:2]
    lines = [
        f"f{fidx:04d}/{total_f}  NO MESH",
        f"mode={mode_str}  yaw={yaw:+.0f}",
        "per-vertex mesh unavailable (profile/back)",
    ]
    for i, line in enumerate(lines):
        y = 26 + i * 22
        cv2.putText(canvas, line, (10, y), font, 0.52, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), font, 0.52, col, 1, cv2.LINE_AA)
    return canvas


def run():
    t0 = time.time()
    print("[mesh-mouth-v1] Building tessellation edges...")
    tess_edges = build_tess_edges_from_mp()
    print(f"  Tessellation edges: {len(tess_edges)}")

    print("[mesh-mouth-v1] Loading v17 rig NPZ...")
    npz = np.load(RIG_NPZ, allow_pickle=True)
    modes_arr   = npz['mode']
    yaw_arr     = npz['yaw_deg']
    pitch_arr   = npz['pitch_deg']
    roll_arr    = npz['roll_deg']
    hcx_arr     = npz['head_center_px']
    hsc_arr     = npz['head_scale_px']
    bs_arr      = npz['blendshapes']   # (847, 52)
    arkit_names = list(npz['arkit_names'])
    total_f     = len(modes_arr)
    print(f"  NPZ: {total_f} frames, v={npz['pipeline_version'][0]}")

    # ARKit indices for HUD + analysis
    jaw_open_idx  = arkit_names.index('jawOpen')
    mouth_close_idx = arkit_names.index('mouthClose')
    smile_l_idx   = arkit_names.index('mouthSmileLeft')
    blink_l_idx   = arkit_names.index('eyeBlinkLeft')
    brow_idx      = arkit_names.index('browInnerUp')

    print("[mesh-mouth-v1] Init MediaPipe FaceLandmarker (blendshapes=True)...")
    face_lmk = make_face_landmarker_with_blendshapes()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {total_f} frames @ {fps:.1f}fps  {fw}x{fh}")

    # Stream storage
    lm_norm_store  = np.full((total_f, 478, 3), np.nan, dtype=np.float32)  # normalized
    lm_px_store    = np.full((total_f, 478, 2), np.nan, dtype=np.float32)  # pixel
    bs_live_store  = np.full((total_f, 52), np.nan, dtype=np.float32)      # live blendshapes
    mesh_quality   = np.zeros(total_f, dtype=np.float32)                    # 0.0 or 1.0
    mode_out       = np.full(total_f, 'NO_MESH', dtype='<U12')

    # Montage frame targets
    # Selected from blendshape analysis:
    # f629: mouth CLOSED, frontal
    # f087: mouth opening (jawOpen=0.184)
    # f485: mouth OPEN MAX (jawOpen=0.578)
    # f251: smile + jaw (jawOpen=0.215, smileL=0.197)
    # f665: neutral frontal closed (for contrast)
    montage_targets = {
        629: 'closed',
        87:  'opening',
        485: 'open-max',
        251: 'smile',
        665: 'neutral',
    }
    saved_montage = {}    # label → (canvas, fidx)

    # Per-frame report
    frame_report = []

    # Video writer
    tmp_path = MASTER_PATH.replace('.mp4', '_tmp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (fw, fh))

    n_primary  = 0
    n_fallback = 0
    n_mp_in_npz = 0  # NPZ mode=MEDIAPIPE frames
    n_mp_detected = 0  # live detection succeeded

    print("[mesh-mouth-v1] Rendering...")
    for fidx in range(total_f):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        npz_mode  = str(modes_arr[fidx])
        yaw       = float(yaw_arr[fidx])
        pitch     = float(pitch_arr[fidx])
        roll      = float(roll_arr[fidx])
        cx_px     = float(hcx_arr[fidx, 0])
        cy_px     = float(hcx_arr[fidx, 1])
        sc_px     = float(hsc_arr[fidx])
        bs_npz    = bs_arr[fidx]   # (52,) from v17 NPZ

        if npz_mode == 'MEDIAPIPE':
            n_mp_in_npz += 1

        canvas = frame_bgr.copy()

        # ── Try live MediaPipe detection ───────────────────────────────────
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        mp_result = face_lmk.detect(mp_img)

        mesh_accepted = False
        L_norm = None
        bs_live = bs_npz.copy()   # default: use NPZ blendshapes

        if mp_result.face_landmarks:
            pts = mp_result.face_landmarks[0]
            L_norm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)

            # Anchor gate: face must be within 300px of NPZ anchor
            det_cx = float(np.mean(L_norm[:, 0]) * fw)
            det_cy = float(np.mean(L_norm[:, 1] * fh))
            dist = math.sqrt((det_cx - cx_px)**2 + (det_cy - cy_px)**2)
            max_dist = max(300.0, sc_px * 2.5)

            if dist <= max_dist:
                mesh_accepted = True
                n_mp_detected += 1

                # Extract live blendshapes (52 ARKit)
                if mp_result.face_blendshapes:
                    bs_cats = mp_result.face_blendshapes[0]
                    for j, cat in enumerate(bs_cats):
                        if j < 52:
                            bs_live[j] = float(cat.score)

        if mesh_accepted and L_norm is not None:
            # PRIMARY: draw full mesh
            canvas = draw_mesh_primary(canvas, L_norm, fw, fh, tess_edges,
                                       bs_live, arkit_names)

            # Blendshape HUD (top-right)
            canvas = draw_blendshape_hud(canvas, bs_live, arkit_names,
                                         jaw_open_idx, smile_l_idx,
                                         blink_l_idx, brow_idx, fw, fh)

            # Mouth zoom inset (bottom-right)
            canvas = draw_mouth_zoom_inset(canvas, L_norm, fw, fh, tess_edges)

            # HUD
            canvas = draw_hud_primary(canvas, fidx, total_f, yaw, pitch, roll,
                                      float(bs_live[jaw_open_idx]), npz_mode)

            # Store stream
            lm_norm_store[fidx] = L_norm
            lm_px_store[fidx] = np.stack([L_norm[:, 0] * fw,
                                            L_norm[:, 1] * fh], axis=1)
            bs_live_store[fidx] = bs_live
            mesh_quality[fidx]  = 1.0
            mode_out[fidx]      = 'MEDIAPIPE'
            n_primary += 1

        else:
            # FALLBACK: no per-vertex mesh
            canvas = draw_hud_fallback(canvas, fidx, total_f, yaw, pitch, roll, npz_mode)
            # Store NPZ blendshapes (decayed, no live mesh)
            bs_live_store[fidx] = bs_npz
            mesh_quality[fidx]  = 0.0
            mode_out[fidx]      = 'NO_MESH'
            n_fallback += 1

        writer.write(canvas)

        # Montage
        if fidx in montage_targets:
            label = montage_targets[fidx]
            saved_montage[label] = (canvas.copy(), fidx)

        # Per-frame report entry
        frame_report.append({
            'frame': fidx,
            'npz_mode': npz_mode,
            'mesh_mode': str(mode_out[fidx]),
            'jawOpen': float(bs_live_store[fidx, jaw_open_idx]),
            'mouthClose': float(bs_live_store[fidx, mouth_close_idx]),
            'mouthSmileLeft': float(bs_live_store[fidx, smile_l_idx]),
            'eyeBlinkLeft': float(bs_live_store[fidx, blink_l_idx]),
            'browInnerUp': float(bs_live_store[fidx, brow_idx]),
            'yaw_deg': yaw,
        })

        if fidx % 100 == 0:
            elapsed = time.time() - t0
            fps_proc = (fidx + 1) / max(elapsed, 0.01)
            print(f"  f{fidx}/{total_f}: primary={n_primary} fallback={n_fallback}  {fps_proc:.1f}fps")

    cap.release()
    writer.release()
    face_lmk.close()

    # ── Encode videos ─────────────────────────────────────────────────────────
    print("\n[mesh-mouth-v1] Encoding master H.264...")
    subprocess.run([
        'ffmpeg', '-y', '-i', tmp_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', MASTER_PATH
    ], check=True, capture_output=True)
    os.remove(tmp_path)
    master_mb = os.path.getsize(MASTER_PATH) / 1e6

    print("[mesh-mouth-v1] Encoding preview (<8MB)...")
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

    # ── Montage: 5 frames ─────────────────────────────────────────────────────
    print("[mesh-mouth-v1] Building mouth-movement montage...")
    order = ['closed', 'opening', 'open-max', 'smile', 'neutral']
    label_text = {
        'closed':   'CLOSED f629 jaw=0.000',
        'opening':  'OPENING f087 jaw=0.184',
        'open-max': 'MAX-OPEN f485 jaw=0.578',
        'smile':    'SMILE f251 jaw=0.215',
        'neutral':  'NEUTRAL f665 jaw=0.000',
    }
    cells = []
    cell_w, cell_h = 360, 640
    for label in order:
        if label in saved_montage:
            frame, fi = saved_montage[label]
            cell = cv2.resize(frame, (cell_w, cell_h))
            # Bottom label
            txt = label_text.get(label, label)
            cv2.rectangle(cell, (0, cell_h - 30), (cell_w, cell_h), (0, 0, 0), -1)
            cv2.putText(cell, txt, (4, cell_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
            cells.append(cell)

    if cells:
        montage = np.concatenate(cells, axis=1)
        cv2.imwrite(MONTAGE_PATH, montage)
        print(f"  Montage: {len(cells)} frames → {MONTAGE_PATH}")

    # ── Save mesh stream NPZ ──────────────────────────────────────────────────
    print("[mesh-mouth-v1] Saving mesh+blendshape stream...")
    np.savez_compressed(
        STREAM_PATH,
        landmarks_478   = lm_norm_store,      # (N, 478, 3) normalized; NaN on NO_MESH
        landmarks_px    = lm_px_store,        # (N, 478, 2) pixel coords; NaN on NO_MESH
        blendshapes_52  = bs_live_store,      # (N, 52) live ARKit; decayed on NO_MESH
        arkit_names     = np.array(arkit_names, dtype='<U25'),
        frame           = np.arange(total_f, dtype=np.int32),
        mode            = mode_out,           # 'MEDIAPIPE' or 'NO_MESH'
        jawOpen         = bs_live_store[:, jaw_open_idx],
        mouthClose      = bs_live_store[:, mouth_close_idx],
        mesh_quality    = mesh_quality,       # 1.0=full mesh; 0.0=no mesh
        yaw_deg         = yaw_arr,
        pitch_deg       = pitch_arr,
        roll_deg        = roll_arr,
        pipeline_version = np.array(['mesh_mouth_v1']),
    )
    stream_kb = os.path.getsize(STREAM_PATH) / 1024
    print(f"  Stream: {STREAM_PATH}  ({stream_kb:.0f} KB)")

    # ── Report ────────────────────────────────────────────────────────────────
    # Compute mouth-movement evidence: jawOpen variance on PRIMARY frames
    mp_mask = mesh_quality > 0.5
    jaw_mp = bs_live_store[mp_mask, jaw_open_idx]
    jaw_range = float(jaw_mp.max() - jaw_mp.min()) if len(jaw_mp) > 0 else 0.0
    jaw_std   = float(jaw_mp.std()) if len(jaw_mp) > 0 else 0.0

    # Multi-frame evidence: consecutive pairs with significant jaw change
    jaw_all = bs_live_store[:, jaw_open_idx]
    big_move_pairs = 0
    for i in range(total_f - 1):
        if mesh_quality[i] > 0.5 and mesh_quality[i+1] > 0.5:
            if abs(float(jaw_all[i+1]) - float(jaw_all[i])) > 0.02:
                big_move_pairs += 1

    report = {
        'pipeline_version': 'mesh_mouth_v1',
        'total_frames': total_f,
        'primary_frames': int(n_primary),
        'primary_pct': round(100 * n_primary / total_f, 1),
        'fallback_frames': int(n_fallback),
        'fallback_pct': round(100 * n_fallback / total_f, 1),
        'np_in_npz': int(n_mp_in_npz),
        'np_detected_live': int(n_mp_detected),
        'mouth_movement_evidence': {
            'jawOpen_max_primary': float(jaw_mp.max()) if len(jaw_mp) else 0.0,
            'jawOpen_min_primary': float(jaw_mp.min()) if len(jaw_mp) else 0.0,
            'jawOpen_range_primary': jaw_range,
            'jawOpen_std_primary': jaw_std,
            'consecutive_pairs_jaw_change_gt_0p02': big_move_pairs,
            'verdict': (
                'CONFIRMED' if jaw_range > 0.2 and big_move_pairs > 5
                else 'PARTIAL' if jaw_range > 0.05
                else 'NOT_DETECTED'
            ),
        },
        'montage_frames': {k: v[1] for k, v in saved_montage.items()},
        'outputs': {
            'master': MASTER_PATH,
            'preview': PREVIEW_PATH,
            'montage': MONTAGE_PATH,
            'stream': STREAM_PATH,
        },
        'output_sizes_mb': {
            'master': round(master_mb, 2),
            'preview': round(preview_mb, 2),
            'stream_kb': round(stream_kb, 0),
        },
        'frames': frame_report,
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {REPORT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print("MESH + MOUTH MOVEMENT V1 — SUMMARY")
    print(f"{'='*65}")
    print(f"Total frames:          {total_f}")
    print(f"PRIMARY (mesh live):   {n_primary}  ({100*n_primary/total_f:.1f}%)")
    print(f"  MP in NPZ:           {n_mp_in_npz}")
    print(f"  MP detected live:    {n_mp_detected}")
    print(f"FALLBACK (no mesh):    {n_fallback}  ({100*n_fallback/total_f:.1f}%)")
    print(f"jawOpen range (MP):    {jaw_range:.4f}")
    print(f"jawOpen std (MP):      {jaw_std:.4f}")
    print(f"Consec jaw moves >0.02:{big_move_pairs}")
    print(f"Mouth verdict:         {report['mouth_movement_evidence']['verdict']}")
    print(f"Master:   {MASTER_PATH}  ({master_mb:.1f} MB)")
    print(f"Preview:  {PREVIEW_PATH}  ({preview_mb:.1f} MB)")
    print(f"Montage:  {MONTAGE_PATH}")
    print(f"Stream:   {STREAM_PATH}  ({stream_kb:.0f} KB)")
    print(f"Time:     {elapsed:.0f}s")
    print(f"{'='*65}")
    return report


if __name__ == '__main__':
    run()
