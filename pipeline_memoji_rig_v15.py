#!/usr/bin/env python3
"""
pipeline_memoji_rig_v15.py — v15 rig stream: 2 principled fixes on v14.

CHANGE 1 — EAR-MIDPOINT ANCHOR UNIFICATION (Proposal 2 from RESEARCH_INNOVATE_TRACKING.md):
  Root cause fixed: v14's face_head_anchor_hybrid() used 3D projection (FOCAL_LEN=700) for
  cx/cy during MediaPipe frames, with a pose cross-gate fallback for cy when projection was
  >120px off.  This still left:
    - A systematic frontal offset (~40px) because nose-tip projection ≠ ear-midpoint
    - A forehead shift in f820-844 (close-up) where the fallback placed anchor at ear/forehead
      rather than nose
  Fix: for ALL MediaPipe-face frames, compute the anchor as the midpoint of face landmarks
  234 (left ear tragion) and 454 (right ear tragion) in the 478-pt mesh, projected to pixel
  space via the same normalized coordinate system MediaPipe uses.  This makes the face-anchor
  and the pose-anchor agree on the SAME anatomical reference (ear-midpoint) across all frames,
  eliminating the reference-point mismatch that is the root cause of the systematic offset.
  The yaw-conditioned calibration is now trained on ear-midpoint face anchors vs pose ear
  anchors, so the residual should be much smaller.

CHANGE 2 — SOURCE-AWARE HETEROSCEDASTIC KALMAN R (Proposal 1 from RESEARCH_INNOVATE_TRACKING.md):
  Root cause fixed: the RTS smoother used a constant r_mp/r_pose_base for ALL anchor sources,
  which caused the backward pass to weight low-confidence pose_calib and pose_raw measurements
  equally with high-confidence mediapipe_face measurements.  The result was the f799-844 Zone 2
  overshoot (backward pass smearing a mode-switch discontinuity into preceding frames).
  Fix: assign per-frame observation noise R_t based on anchor_source:
    R_mp_face   = 15² = 225    (ear-midpoint directly from face landmarks, high precision)
    R_pose_calib= 45² = 2025   (calibrated pose, RMSE ~28-35px → ~45px per axis)
    R_pose_raw  = 80² = 6400   (raw pose, no calibration, low trust)
    R_hold      = 500² = 250000 (no detection, near-infinite — let Kalman predict freely)
  The smoother now weights each measurement by its true reliability and will not violently
  pull high-confidence neighboring frames toward a noisy mode-switch measurement.

All other code is identical to v14.
Outputs: memoji_rig_stream_v13.npz (OVERWRITES previous — tracking_confirm_overlay.py
reads this same file).

Python: python3
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
import trimesh

import torch
import torch.nn as nn
from torchvision import models, transforms

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH       = "input_clip.mov"
FACE_MODEL_TASK  = "models/face_landmarker.task"
POSE_MODEL_TASK  = "models/pose_landmarker_full.task"
CANONICAL_OBJ    = "assets/canonical_face_model.obj"
YOLO_MODEL_PATH  = "models/yolov10n-face.pt"
REP360_WEIGHTS   = "models/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
OUT_DIR          = "."

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"[v15-rig] Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BSHP_DECAY   = 0.92
BSHP_NEUTRAL = 0.0
YAW_SIGN     = -1.0
PITCH_SIGN   = -1.0
FOCAL_LEN    = 700.0
VIS_THRESH   = 0.30          # pose landmark visibility threshold
MIN_EAR_SPAN = 10.0          # px; below this ear-based scale is unreliable

# FIX 1 (v14): jump-gate threshold (pixels)
JUMP_GATE_PX = 150.0

# FIX 2 (v14): RTS backward-vs-forward fallback threshold (pixels)
RTS_MAX_DEV_PX = 100.0

# FIX 3 (v14): scale discontinuity threshold for segment split (pixels)
SCALE_JUMP_RESET_PX = 80.0

# CHANGE 1 (v15): MediaPipe 478-pt mesh ear-tragion landmark indices
# 234 = left ear tragion, 454 = right ear tragion (standard MediaPipe face mesh topology)
MP_FACE_EAR_LEFT_IDX  = 234
MP_FACE_EAR_RIGHT_IDX = 454

# CHANGE 2 (v15): per-source observation noise R values (variance = sigma²)
# These are per-AXIS (x and y independently); the smoother uses a scalar per frame
R_FACE_EAR    = 15.0 ** 2    # 225   — ear-midpoint direct from face landmarks
R_POSE_CALIB  = 45.0 ** 2    # 2025  — calibrated pose anchor (RMSE ~28-35px → 45px budget)
R_POSE_RAW    = 80.0 ** 2    # 6400  — raw pose anchor, no calibration
R_HOLD        = 500.0 ** 2   # 250000 — no detection: near-infinite, Kalman predicts freely

ARKIT_NAMES = [
    "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
    "cheekPuff","cheekSquintLeft","cheekSquintRight",
    "eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft","eyeLookDownRight",
    "eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight",
    "eyeLookUpLeft","eyeLookUpRight","eyeSquintLeft","eyeSquintRight",
    "eyeWideLeft","eyeWideRight",
    "jawForward","jawLeft","jawOpen","jawRight",
    "mouthClose","mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight",
    "mouthFunnel","mouthLeft","mouthLowerDownLeft","mouthLowerDownRight",
    "mouthPressLeft","mouthPressRight","mouthPucker","mouthRight",
    "mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
    "mouthSmileLeft","mouthSmileRight","mouthStretchLeft","mouthStretchRight",
    "mouthUpperUpLeft","mouthUpperUpRight",
    "noseSneerLeft","noseSneerRight",
    "tongueOut",
]

IMG_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# 6DRepNet360 (unchanged from v14)
# ─────────────────────────────────────────────────────────────────────────────
class SixDRepNet360(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.conv1 = backbone.conv1; self.bn1 = backbone.bn1
        self.relu  = backbone.relu;  self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1; self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3; self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.linear_reg = nn.Linear(2048, 6)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.linear_reg(x)


def ortho6d_to_R(out6d: np.ndarray) -> np.ndarray:
    a1 = out6d[:3]; a2 = out6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= (np.linalg.norm(b2) + 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def run_rep360_R(model: SixDRepNet360, frame_bgr: np.ndarray,
                 box_xyxy: List[float]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = frame_bgr.shape[:2]
    pad = int(0.10 * max(x2-x1, y2-y1))
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(w, x2+pad); y2 = min(h, y2+pad)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    inp = IMG_TRANSFORM(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out6d = model(inp)[0].cpu().numpy()
    R_raw = ortho6d_to_R(out6d)
    e = Rotation.from_matrix(R_raw).as_euler('YXZ', degrees=True)
    e_corr = np.array([YAW_SIGN * e[0], PITCH_SIGN * e[1], e[2]])
    return Rotation.from_euler('YXZ', e_corr, degrees=True).as_matrix()


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe FaceLandmarker
# ─────────────────────────────────────────────────────────────────────────────
def make_mp_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_TASK),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        min_tracking_confidence=0.2,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_TASK),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def extract_mp_result(result) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    T, B, L = None, None, None
    if result.facial_transformation_matrixes:
        T = np.array(result.facial_transformation_matrixes[0])
    if result.face_blendshapes:
        B = np.array([c.score for c in result.face_blendshapes[0]], dtype=np.float32)
    if result.face_landmarks:
        pts = result.face_landmarks[0]
        L = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
    return T, B, L


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1: Ear-midpoint anchor from MediaPipe face landmarks
# ─────────────────────────────────────────────────────────────────────────────
def face_ear_midpoint_anchor(L_mp: np.ndarray, fw: int, fh: int) -> Optional[Tuple[float, float]]:
    """
    CHANGE 1 (v15): Extract the ear-midpoint anchor from MediaPipe 478-pt face mesh.

    Landmarks 234 (left ear tragion) and 454 (right ear tragion) are stable anatomical
    points present in the MediaPipe face mesh. Their midpoint is the same anatomical
    reference used by the pose landmarker ear anchor, so using this eliminates the
    reference-point mismatch between mediapipe_face and pose_calib/pose_raw anchors.

    MediaPipe normalizes landmark coordinates to [0,1] range (x=left→right, y=top→bottom).
    Pixel conversion: cx_px = x * fw, cy_px = y * fh.

    Returns (cx_px, cy_px) or None if landmarks not available or out of bounds.
    """
    if L_mp is None:
        return None
    n_lm = L_mp.shape[0]
    if n_lm <= max(MP_FACE_EAR_LEFT_IDX, MP_FACE_EAR_RIGHT_IDX):
        # 478-pt mesh not available (shouldn't happen with FaceLandmarker, but guard)
        return None

    l_ear = L_mp[MP_FACE_EAR_LEFT_IDX]   # [x_norm, y_norm, z_norm]
    r_ear = L_mp[MP_FACE_EAR_RIGHT_IDX]

    # Normalized [0,1] to pixel
    l_ear_px = (l_ear[0] * fw, l_ear[1] * fh)
    r_ear_px = (r_ear[0] * fw, r_ear[1] * fh)

    cx_px = 0.5 * (l_ear_px[0] + r_ear_px[0])
    cy_px = 0.5 * (l_ear_px[1] + r_ear_px[1])

    # Sanity: the ear midpoint should be within the frame (allow small margin)
    margin = 0.1 * max(fw, fh)
    if not (-margin <= cx_px <= fw + margin and -margin <= cy_px <= fh + margin):
        return None

    return (cx_px, cy_px)


# ─────────────────────────────────────────────────────────────────────────────
# Pose-based head anchor extraction (unchanged from v14)
# ─────────────────────────────────────────────────────────────────────────────
def pose_head_anchor(result, fw: int, fh: int) -> Optional[Dict]:
    """
    Extract head anchor from PoseLandmarker result.
    Returns dict with cx_px, cy_px, scale_px, source, confidence
    or None if pose not detected.
    """
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    nose  = lms[0]
    l_ear = lms[7]
    r_ear = lms[8]
    l_sho = lms[11]
    r_sho = lms[12]

    def lm_px(lm):
        return (lm.x * fw, lm.y * fh)

    def lm_vis(lm):
        return lm.visibility >= VIS_THRESH

    nose_px  = lm_px(nose)
    l_ear_px = lm_px(l_ear)
    r_ear_px = lm_px(r_ear)
    l_sho_px = lm_px(l_sho)
    r_sho_px = lm_px(r_sho)

    l_ear_ok = lm_vis(l_ear)
    r_ear_ok = lm_vis(r_ear)
    l_sho_ok = lm_vis(l_sho)
    r_sho_ok = lm_vis(r_sho)
    nose_ok  = lm_vis(nose)

    if l_ear_ok and r_ear_ok:
        cx = 0.5 * (l_ear_px[0] + r_ear_px[0])
        cy = 0.5 * (l_ear_px[1] + r_ear_px[1])
        ear_span = math.sqrt((l_ear_px[0]-r_ear_px[0])**2 + (l_ear_px[1]-r_ear_px[1])**2)
        scale = ear_span if ear_span >= MIN_EAR_SPAN else None
        confidence = min(l_ear.visibility, r_ear.visibility)
        source = 'pose_both_ears'
    elif l_ear_ok:
        cx = l_ear_px[0]
        cy = l_ear_px[1]
        confidence = l_ear.visibility * 0.7
        source = 'pose_left_ear'
        scale = None
    elif r_ear_ok:
        cx = r_ear_px[0]
        cy = r_ear_px[1]
        confidence = r_ear.visibility * 0.7
        source = 'pose_right_ear'
        scale = None
    elif nose_ok:
        cx = nose_px[0]
        cy = nose_px[1]
        confidence = nose.visibility * 0.5
        source = 'pose_nose'
        scale = None
    else:
        return None

    if scale is None:
        if l_sho_ok and r_sho_ok:
            sho_span = math.sqrt((l_sho_px[0]-r_sho_px[0])**2 + (l_sho_px[1]-r_sho_px[1])**2)
            scale = sho_span * 0.45
        else:
            scale = 80.0

    return {
        'cx': cx, 'cy': cy,
        'scale': scale,
        'source': source,
        'confidence': float(confidence),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Canonical mesh utilities (unchanged from v14)
# ─────────────────────────────────────────────────────────────────────────────
def load_canonical_mesh() -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(CANONICAL_OBJ, force='mesh')
    return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int32)


def build_head_transform_from_R(R: np.ndarray, box_xyxy: List[float],
                                 frame_shape: Tuple[int, int],
                                 canonical_scale: float) -> np.ndarray:
    h, w = frame_shape
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    box_w = x2 - x1; box_h = y2 - y1
    scale = max(box_h, box_w) / canonical_scale
    tx = (cx - w/2) / (w/2) * scale * canonical_scale * 0.5
    ty = -(cy - h/2) / (h/2) * scale * canonical_scale * 0.5
    tz = -400.0 * scale
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R * scale
    T[:3,  3] = [tx, ty, tz]
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Head state (unchanged from v14)
# ─────────────────────────────────────────────────────────────────────────────
class HeadState:
    def __init__(self, canon_verts: np.ndarray, canon_faces: np.ndarray):
        self.canon_verts = canon_verts
        self.canon_faces = canon_faces
        vmin = canon_verts.min(axis=0); vmax = canon_verts.max(axis=0)
        self.canon_scale = max(vmax - vmin)
        self.head_transform: np.ndarray = np.eye(4)
        self.blendshapes: np.ndarray = np.zeros(52, dtype=np.float32)
        self.last_yolo_box: Optional[List[float]] = None
        self.frames_since_mp: int = 0

    def update_mp(self, T: np.ndarray, bshps: np.ndarray, *_):
        self.head_transform = T.copy()
        self.blendshapes    = bshps.copy()
        self.frames_since_mp = 0

    def update_rep360(self, R: np.ndarray, box_xyxy: List[float],
                      frame_shape: Tuple[int, int]):
        self.last_yolo_box = box_xyxy
        T = build_head_transform_from_R(R, box_xyxy, frame_shape, self.canon_scale)
        self.head_transform = T.copy()
        self.blendshapes = self.blendshapes * BSHP_DECAY
        self.frames_since_mp += 1

    def update_hold(self):
        self.blendshapes = self.blendshapes * BSHP_DECAY
        self.frames_since_mp += 1


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — forward inference
# CHANGE 1 applied here: face_cx/cy uses ear-midpoint, not 3D projection
# ─────────────────────────────────────────────────────────────────────────────
def forward_pass(cap: cv2.VideoCapture, fw: int, fh: int,
                 face_lmk, pose_lmk,
                 yolo_face: YOLO, rep360: SixDRepNet360,
                 state: HeadState, total_f: int) -> List[Dict]:
    records = []
    n_mp = 0; n_rep360 = 0; n_hold = 0
    n_ear_ok = 0; n_ear_fallback = 0
    t0 = time.time()

    for fidx in range(total_f):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        mp_result  = face_lmk.detect(mp_img)
        T_mp, B_mp, L_mp = extract_mp_result(mp_result)
        has_mp = T_mp is not None and B_mp is not None

        pose_result = pose_lmk.detect(mp_img)
        pa = pose_head_anchor(pose_result, fw, fh)

        yolo_box = None
        if has_mp:
            state.update_mp(T_mp, B_mp, L_mp, (fh, fw))
            mode = 'MEDIAPIPE'; n_mp += 1

            # CHANGE 1: use ear-midpoint from face landmarks as anchor
            ear_mid = face_ear_midpoint_anchor(L_mp, fw, fh)
            if ear_mid is not None:
                face_cx, face_cy = ear_mid
                n_ear_ok += 1
            else:
                # Fallback: if ear landmarks not available, use pose anchor if present
                # (should be very rare — 478-pt mesh always has indices 234+454)
                if pa is not None:
                    face_cx, face_cy = pa['cx'], pa['cy']
                else:
                    face_cx, face_cy = fw / 2.0, fh / 2.0
                n_ear_fallback += 1
                if fidx % 50 == 0 or n_ear_fallback <= 5:
                    print(f"  [CHANGE1-ear] f{fidx}: ear landmarks unavailable — falling back to pose/center")
        else:
            yolo_res = yolo_face(frame_bgr, verbose=False, conf=0.25, device=str(DEVICE))
            boxes = yolo_res[0].boxes
            if boxes is not None and len(boxes) > 0:
                bi = boxes.conf.argmax().item()
                yolo_box = boxes.xyxy[bi].tolist()
                R = run_rep360_R(rep360, frame_bgr, yolo_box)
                if R is not None:
                    state.update_rep360(R, yolo_box, (fh, fw))
                    mode = 'REP360'; n_rep360 += 1
                else:
                    state.update_hold(); mode = 'HOLD'; n_hold += 1
            else:
                state.update_hold(); mode = 'HOLD'; n_hold += 1
            face_cx = face_cy = None

        T = state.head_transform
        R3 = T[:3, :3]
        col_norm = np.linalg.norm(R3[:, 0])
        if col_norm > 1e-6:
            R3_unit = R3 / col_norm
        else:
            R3_unit = R3.copy()
        euler = Rotation.from_matrix(R3_unit).as_euler('YXZ', degrees=True)
        yaw, pitch, roll = euler[0], euler[1], euler[2]

        rec = {
            'frame':         fidx,
            'mode':          mode,
            'head_transform': T.copy(),
            'blendshapes':    state.blendshapes.copy(),
            'yaw_deg':        float(yaw),
            'pitch_deg':      float(pitch),
            'roll_deg':       float(roll),
            'pose_anchor':    pa,
            'face_cx':        face_cx,
            'face_cy':        face_cy,
            'yolo_box':       yolo_box,
        }
        records.append(rec)

        if fidx % 100 == 0:
            e = time.time() - t0
            fps_p = (fidx+1) / max(e, 0.01)
            print(f"  [fwd] f{fidx}/{total_f}: MP={n_mp} REP360={n_rep360} HOLD={n_hold} "
                  f"ear_ok={n_ear_ok} ear_fb={n_ear_fallback}  {fps_p:.1f}fps")

    print(f"  [fwd] Done: MP={n_mp} REP360={n_rep360} HOLD={n_hold} "
          f"ear_ok={n_ear_ok} ear_fallback={n_ear_fallback}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2A — Yaw-conditioned calibration (updated for ear-midpoint reference)
# ─────────────────────────────────────────────────────────────────────────────
def fit_yaw_calibration(records: List[Dict]) -> Optional[Dict]:
    """
    Fit yaw-conditioned linear calibration mapping pose anchor → face ear-midpoint anchor.
    In v15, face_cx/face_cy is the ear-midpoint (same anatomical reference as pose ear),
    so the calibration residual should be significantly smaller than v14's 35-67px.
    """
    rows_x, rows_y, targets_x, targets_y = [], [], [], []
    for r in records:
        if r['mode'] != 'MEDIAPIPE':
            continue
        if r['pose_anchor'] is None:
            continue
        if r['face_cx'] is None:
            continue
        yaw_rad = math.radians(r['yaw_deg'])
        row = [1.0, math.sin(yaw_rad), math.cos(yaw_rad)]
        rows_x.append(row); targets_x.append(r['face_cx'] - r['pose_anchor']['cx'])
        rows_y.append(row); targets_y.append(r['face_cy'] - r['pose_anchor']['cy'])

    if len(rows_x) < 10:
        print(f"  [calib] Only {len(rows_x)} calibration points — skipping")
        return None

    X = np.array(rows_x, dtype=np.float64)
    ax, _, _, _ = np.linalg.lstsq(X, np.array(targets_x), rcond=None)
    bx, _, _, _ = np.linalg.lstsq(X, np.array(targets_y), rcond=None)

    pred_x = X @ ax; pred_y = X @ bx
    res_x = np.array(targets_x) - pred_x
    res_y = np.array(targets_y) - pred_y
    rmse_x = float(np.sqrt(np.mean(res_x**2)))
    rmse_y = float(np.sqrt(np.mean(res_y**2)))
    print(f"  [calib] Fitted on {len(rows_x)} MP frames: "
          f"a=[{ax[0]:.1f},{ax[1]:.1f},{ax[2]:.1f}] RMSE_x={rmse_x:.1f}px  "
          f"b=[{bx[0]:.1f},{bx[1]:.1f},{bx[2]:.1f}] RMSE_y={rmse_y:.1f}px")
    return {
        'ax': ax.tolist(), 'bx': bx.tolist(),
        'n_points': len(rows_x),
        'rmse_x_px': rmse_x, 'rmse_y_px': rmse_y,
    }


def apply_calibration(pose_cx: float, pose_cy: float,
                       yaw_deg: float, calib: Optional[Dict],
                       mode: str = 'REP360') -> Tuple[float, float]:
    if calib is None:
        return pose_cx, pose_cy
    if mode == 'HOLD' or abs(yaw_deg) > 80.0:
        return pose_cx, pose_cy
    yaw_rad = math.radians(yaw_deg)
    row = np.array([1.0, math.sin(yaw_rad), math.cos(yaw_rad)])
    dx = float(np.dot(row, calib['ax']))
    dy = float(np.dot(row, calib['bx']))
    dx = float(np.clip(dx, -50.0, 50.0))
    dy = float(np.clip(dy, -50.0, 50.0))
    return pose_cx + dx, pose_cy + dy


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2B — Compute per-frame raw anchors WITH FIX 1 (jump-gate, v14) + CHANGE 1 (ear)
# ─────────────────────────────────────────────────────────────────────────────
def compute_raw_anchors(records: List[Dict],
                        calib: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Determine (cx_raw, cy_raw, scale_raw, anchor_source, anchor_confidence) per frame.

    FIX 1 (v14) — JUMP-GATE: reject MP anchor if it moves >150px from previous in one frame.
    CHANGE 1 (v15) — ANCHOR: face_cx/face_cy is now the ear-midpoint (not 3D projection).

    anchor_source values:
      'mediapipe_face'   — ear-midpoint from MediaPipe 478-pt face mesh
      'pose_calib(...)'  — calibrated pose anchor
      'pose_raw(...)'    — raw pose anchor (extreme yaw or HOLD)
      'predicted'        — no measurement, smoother predicts
    """
    N = len(records)
    cx_raw  = np.full(N, np.nan, dtype=np.float64)
    cy_raw  = np.full(N, np.nan, dtype=np.float64)
    sc_raw  = np.full(N, np.nan, dtype=np.float64)
    sources = ['unknown'] * N
    confs   = np.zeros(N, dtype=np.float64)

    # Track the last accepted anchor for the jump-gate
    prev_cx: Optional[float] = None
    prev_cy: Optional[float] = None
    n_jump_rejected = 0

    for i, r in enumerate(records):
        pa = r['pose_anchor']

        if r['mode'] == 'MEDIAPIPE' and r['face_cx'] is not None:
            face_cx = r['face_cx']   # v15: this is the ear-midpoint
            face_cy = r['face_cy']

            # FIX 1 (v14): jump-gate check
            jump_ok = True
            if prev_cx is not None:
                dist = math.sqrt((face_cx - prev_cx)**2 + (face_cy - prev_cy)**2)
                if dist > JUMP_GATE_PX:
                    jump_ok = False
                    n_jump_rejected += 1
                    print(f"  [FIX1-jumpgate] f{i}: MP ear-mid jump {dist:.1f}px > {JUMP_GATE_PX}px — REJECTED")

            if jump_ok:
                # Accept the mediapipe ear-midpoint anchor
                cx_raw[i] = face_cx
                cy_raw[i] = face_cy
                # Scale: use ear span if pose anchor is available and has ear info,
                # otherwise fall back to pose scale if pose was tracking
                sc_raw[i] = pa['scale'] if pa is not None else 80.0
                sources[i] = 'mediapipe_face'
                confs[i]   = 1.0
                prev_cx = face_cx
                prev_cy = face_cy
            else:
                # Jump-gate rejected: fall back to pose anchor
                if pa is not None:
                    pcx, pcy = apply_calibration(pa['cx'], pa['cy'], r['yaw_deg'], calib,
                                                 mode=r['mode'])
                    cx_raw[i] = pcx
                    cy_raw[i] = pcy
                    sc_raw[i] = pa['scale']
                    mode_tag   = 'calib' if abs(r['yaw_deg']) <= 80 else 'raw'
                    sources[i] = f"pose_{mode_tag}_jumpgate({pa['source']})"
                    confs[i]   = pa['confidence'] * 0.85
                    prev_cx = pcx
                    prev_cy = pcy
                else:
                    # No pose either — leave NaN, smoother interpolates
                    sources[i] = 'predicted_jumpgate'
                    confs[i]   = 0.0

        elif pa is not None:
            # Pose anchor + calibration
            pcx, pcy = apply_calibration(pa['cx'], pa['cy'], r['yaw_deg'], calib,
                                         mode=r['mode'])
            cx_raw[i] = pcx
            cy_raw[i] = pcy
            sc_raw[i] = pa['scale']
            mode_tag   = 'calib' if r['mode'] == 'REP360' and abs(r['yaw_deg']) <= 80 else 'raw'
            sources[i] = f"pose_{mode_tag}({pa['source']})"
            confs[i]   = pa['confidence'] * 0.85
            prev_cx = pcx
            prev_cy = pcy
        else:
            # No measurement
            sources[i] = 'predicted'
            confs[i]   = 0.0

    print(f"  [FIX1-jumpgate] Total jump-rejected frames: {n_jump_rejected}")
    return cx_raw, cy_raw, sc_raw, sources, confs


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 (v14): Detect scale discontinuity boundaries for segment splitting
# ─────────────────────────────────────────────────────────────────────────────
def find_scale_discontinuity_boundaries(sc_raw: np.ndarray,
                                         threshold_px: float = SCALE_JUMP_RESET_PX
                                         ) -> List[int]:
    boundaries = []
    valid_idx = np.where(~np.isnan(sc_raw))[0]
    if len(valid_idx) < 2:
        return boundaries

    for k in range(1, len(valid_idx)):
        i_prev = valid_idx[k-1]
        i_curr = valid_idx[k]
        if i_curr - i_prev <= 3:
            jump = abs(sc_raw[i_curr] - sc_raw[i_prev])
            if jump > threshold_px:
                boundaries.append(int(i_curr))
                print(f"  [FIX3-scalereset] Scale jump at f{i_prev}→f{i_curr}: "
                      f"{sc_raw[i_prev]:.1f}→{sc_raw[i_curr]:.1f}px ({jump:.1f}px jump) "
                      f"— adding segment boundary at f{i_curr}")

    return boundaries


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2: per-source R lookup for heteroscedastic Kalman
# ─────────────────────────────────────────────────────────────────────────────
def r_for_source(source: str) -> float:
    """
    CHANGE 2 (v15): Return per-axis observation noise variance R based on anchor_source.

    Calibrated values:
      mediapipe_face  → R = 15² = 225   (ear-midpoint direct from mesh, ~10-20px error)
      pose_calib      → R = 45² = 2025  (calibration RMSE budget ~28-45px per axis)
      pose_raw        → R = 80² = 6400  (raw uncalibrated pose, low trust)
      predicted/hold  → R = 500² = 250000 (no observation — let Kalman predict)

    The source string may contain additional qualifiers (e.g. 'pose_calib(pose_both_ears)')
    so we use startswith() matching.
    """
    src = str(source)
    if src.startswith('mediapipe_face'):
        return R_FACE_EAR
    elif src.startswith('pose_calib'):
        return R_POSE_CALIB
    elif src.startswith('pose_raw'):
        return R_POSE_RAW
    else:
        # 'predicted', 'predicted_jumpgate', 'unknown', or any unmatched source
        return R_HOLD


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2C — Forward-backward Kalman smoother WITH FIX 2 + FIX 3 + CHANGE 2
# ─────────────────────────────────────────────────────────────────────────────
def fb_kalman_smooth(cx_raw: np.ndarray, cy_raw: np.ndarray,
                     sc_raw: np.ndarray,
                     confs: np.ndarray,
                     sources: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-backward Kalman smoother with v14 fixes + CHANGE 2 (heteroscedastic R).

    CHANGE 2: instead of r_mp / max(c, 0.05) as a uniform noise for all frames,
    use per-frame R_t = r_for_source(sources[i]).  The confidence scalar is now
    used only as a secondary multiplier on pose sources (not on mediapipe_face,
    which is already high-confidence by construction).

    FIX 2 (v14): RTS vs forward-only fallback where deviation > RTS_MAX_DEV_PX.
    FIX 3 (v14): segment split at scale discontinuities.
    """
    N = len(cx_raw)

    dt   = 1.0
    q_px = 4.0

    def r_frame(i: int) -> float:
        """Get observation noise R for frame i, incorporating source + confidence."""
        base_r = r_for_source(sources[i] if sources else 'predicted')
        # For pose sources, inflate slightly if confidence is low
        if sources and str(sources[i]).startswith('pose_') and confs[i] > 0.01:
            # conf is in [0,1]; low conf → inflate R up to 2x
            conf_inflate = 1.0 / max(confs[i], 0.3)
            return base_r * conf_inflate
        return base_r

    def run_kalman_segment(meas: np.ndarray, seg_sources: List[str],
                           seg_confs: np.ndarray,
                           start: int, end: int):
        """
        Run forward Kalman on meas[start:end] with per-frame R_t.
        Returns (x_fwd[start:end], P_fwd[start:end], x_pred[start:end], P_pred[start:end]).
        """
        seg_len = end - start
        if seg_len <= 0:
            empty = np.zeros((0, 2))
            return empty, np.zeros((0, 2, 2)), empty, np.zeros((0, 2, 2))

        F = np.array([[1., dt], [0., 1.]])
        H = np.array([[1., 0.]])
        Q = np.array([[q_px*dt**3/3, q_px*dt**2/2],
                      [q_px*dt**2/2, q_px*dt]])

        xs     = np.zeros((seg_len, 2), dtype=np.float64)
        Ps     = np.zeros((seg_len, 2, 2), dtype=np.float64)
        xpreds = np.zeros((seg_len, 2), dtype=np.float64)
        Ppreds = np.zeros((seg_len, 2, 2), dtype=np.float64)

        # Initialise from first valid measurement in segment
        first_valid = None
        for j in range(seg_len):
            gi = start + j
            if not np.isnan(meas[gi]) and seg_confs[gi] > 0.0:
                first_valid = j; break

        # State init
        x = np.zeros(2, dtype=np.float64)
        P = np.eye(2) * 1e4
        if first_valid is not None:
            gi = start + first_valid
            x[0] = meas[gi]; x[1] = 0.0
            # Init P with the first-frame R
            P[0,0] = r_frame(gi)
            P[1,1] = q_px

        for j in range(seg_len):
            gi = start + j
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            xpreds[j] = x_pred
            Ppreds[j] = P_pred

            m = meas[gi]
            c = seg_confs[gi]
            # CHANGE 2: use per-frame heteroscedastic R
            R_t = r_frame(gi)
            if not np.isnan(m) and c >= 0.0 and R_t < R_HOLD:
                # Valid measurement (R_HOLD means no observation — skip update)
                S   = float((H @ P_pred @ H.T)[0, 0] + R_t)
                K   = (P_pred @ H.T).ravel() / S
                innov = float(m - (H @ x_pred)[0])
                x = x_pred + K * innov
                P = (np.eye(2) - np.outer(K, H[0])) @ P_pred
            else:
                x = x_pred
                P = P_pred

            xs[j] = x; Ps[j] = P

        return xs, Ps, xpreds, Ppreds

    def rts_smooth_segment(xs, Ps, xpreds, Ppreds, F):
        """RTS backward pass on a segment."""
        seg_len = len(xs)
        xs_s = xs.copy()
        Ps_s = Ps.copy()
        for j in range(seg_len - 2, -1, -1):
            P_pred_next = Ppreds[j+1]
            try:
                Pinv = np.linalg.inv(P_pred_next)
            except np.linalg.LinAlgError:
                continue
            G = Ps[j] @ F.T @ Pinv
            xs_s[j] = xs[j] + G @ (xs_s[j+1] - xpreds[j+1])
            Ps_s[j] = Ps[j] + G @ (Ps_s[j+1] - P_pred_next) @ G.T
        return xs_s

    F_mat = np.array([[1., dt], [0., 1.]])

    # FIX 3 (v14): find scale discontinuity boundaries
    boundaries = find_scale_discontinuity_boundaries(sc_raw)
    seg_starts = [0] + boundaries
    seg_ends   = boundaries + [N]
    segments   = list(zip(seg_starts, seg_ends))
    print(f"  [FIX3] Smoother segments: {segments}")

    # Build a flat per-frame source list for r_frame() to reference
    # (sources is already global-frame indexed so no offset needed)
    seg_sources = sources if sources else ['predicted'] * N
    seg_confs   = confs

    def smooth_channel(meas, max_pull_px=40.0):
        """
        Run forward Kalman + RTS per segment (FIX 3), then apply forward-only
        fallback where backward deviation exceeds RTS_MAX_DEV_PX (FIX 2).
        CHANGE 2: per-frame R via r_frame().
        """
        fwd_pos  = np.full(N, np.nan, dtype=np.float64)
        rts_pos  = np.full(N, np.nan, dtype=np.float64)

        for (seg_start, seg_end) in segments:
            if seg_start >= seg_end:
                continue
            xs, Ps, xpreds, Ppreds = run_kalman_segment(meas, seg_sources, seg_confs,
                                                         seg_start, seg_end)
            xs_s = rts_smooth_segment(xs, Ps, xpreds, Ppreds, F_mat)

            for j in range(seg_end - seg_start):
                gi = seg_start + j
                fwd_pos[gi] = xs[j, 0]
                rts_pos[gi] = xs_s[j, 0]

        # Combine: start with RTS result
        result = rts_pos.copy()

        # Clamp: if raw measurement exists and smoother moved > max_pull_px, limit
        for i in range(N):
            if not np.isnan(meas[i]) and confs[i] > 0.0:
                raw = meas[i]
                delta = result[i] - raw
                if abs(delta) > max_pull_px:
                    result[i] = raw + np.sign(delta) * max_pull_px

        # FIX 2 (v14): RTS vs forward-only fallback
        n_fallback = 0
        for i in range(N):
            if not np.isnan(fwd_pos[i]) and not np.isnan(result[i]):
                dev = abs(result[i] - fwd_pos[i])
                if dev > RTS_MAX_DEV_PX:
                    result[i] = fwd_pos[i]
                    n_fallback += 1
        if n_fallback > 0:
            print(f"  [FIX2-rts-fallback] {n_fallback} frames fell back to forward-only estimate")

        return result

    cx_smooth = smooth_channel(cx_raw)
    cy_smooth = smooth_channel(cy_raw)
    sc_smooth = smooth_channel(sc_raw)

    # Fill remaining NaN with linear interp
    for arr in [cx_smooth, cy_smooth, sc_smooth]:
        nans = np.isnan(arr)
        if nans.any():
            ok = ~nans
            if ok.sum() >= 2:
                arr[nans] = np.interp(np.where(nans)[0], np.where(ok)[0], arr[ok])
            else:
                arr[nans] = 0.0

    return cx_smooth, cy_smooth, sc_smooth


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run():
    t_start = time.time()
    print("[v15-rig] Loading models...")
    face_lmk  = make_mp_landmarker()
    pose_lmk  = make_pose_landmarker()
    yolo_face = YOLO(YOLO_MODEL_PATH)

    rep360 = SixDRepNet360()
    rep360.load_state_dict(torch.load(REP360_WEIGHTS, map_location='cpu'))
    rep360 = rep360.to(DEVICE)
    rep360.eval()

    canon_verts, canon_faces = load_canonical_mesh()
    state = HeadState(canon_verts, canon_faces)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps     = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[v15-rig] {total_f} frames @ {fps:.1f}fps  ({fw}x{fh})")
    print(f"[v15-rig] CHANGE 1: ear-midpoint anchor (MP face landmarks 234+454)")
    print(f"[v15-rig] CHANGE 2: heteroscedastic Kalman R: mp={R_FACE_EAR:.0f} calib={R_POSE_CALIB:.0f} raw={R_POSE_RAW:.0f} hold={R_HOLD:.0f}")

    print("\n[v15-rig] PASS 1: forward inference...")
    records = forward_pass(cap, fw, fh, face_lmk, pose_lmk,
                           yolo_face, rep360, state, total_f)
    cap.release()
    face_lmk.close()
    pose_lmk.close()

    print("\n[v15-rig] PASS 2A: yaw-conditioned calibration (ear-midpoint reference)...")
    calib = fit_yaw_calibration(records)

    print("[v15-rig] PASS 2B: computing raw anchors (jump-gate + ear-midpoint)...")
    cx_raw, cy_raw, sc_raw, sources, confs = compute_raw_anchors(records, calib)
    n_nan = int(np.isnan(cx_raw).sum())
    src_counts = {}
    for s in sources:
        cat = ('mediapipe_face' if s.startswith('mediapipe_face') else
               'pose_calib' if s.startswith('pose_calib') else
               'pose_raw' if s.startswith('pose_raw') else 'predicted')
        src_counts[cat] = src_counts.get(cat, 0) + 1
    print(f"  anchor_source counts: " + ", ".join(f"{k}={v}" for k, v in src_counts.items()))
    print(f"  NaN anchor frames: {n_nan}")

    print("[v15-rig] PASS 2C: heteroscedastic Kalman smoother (FIX 2 + FIX 3 + CHANGE 2)...")
    cx_smooth, cy_smooth, sc_smooth = fb_kalman_smooth(cx_raw, cy_raw, sc_raw, confs, sources)

    N = len(records)
    frames_arr     = np.array([r['frame'] for r in records], dtype=np.int32)
    modes_arr      = np.array([r['mode']  for r in records])
    transforms_arr = np.array([r['head_transform'] for r in records], dtype=np.float32)
    bshp_arr       = np.array([r['blendshapes'] for r in records], dtype=np.float32)
    yaw_arr        = np.array([r['yaw_deg']   for r in records], dtype=np.float32)
    pitch_arr      = np.array([r['pitch_deg'] for r in records], dtype=np.float32)
    roll_arr       = np.array([r['roll_deg']  for r in records], dtype=np.float32)
    head_center_px = np.stack([cx_smooth, cy_smooth], axis=1).astype(np.float32)
    head_scale_px  = sc_smooth.astype(np.float32)
    anchor_conf    = confs.astype(np.float32)

    # Write NPZ — same filename so tracking_confirm_overlay.py reads it unchanged
    npz_path = f"{OUT_DIR}/memoji_rig_stream_v13.npz"
    np.savez_compressed(
        npz_path,
        frame             = frames_arr,
        mode              = modes_arr,
        head_transform    = transforms_arr,
        blendshapes       = bshp_arr,
        yaw_deg           = yaw_arr,
        pitch_deg         = pitch_arr,
        roll_deg          = roll_arr,
        head_center_px    = head_center_px,
        head_scale_px     = head_scale_px,
        anchor_source     = np.array(sources),
        anchor_confidence = anchor_conf,
        arkit_names       = ARKIT_NAMES,
        pipeline_version  = np.array(['v15']),   # provenance tag
    )
    npz_size = os.path.getsize(npz_path) / 1024
    print(f"\n[v15-rig] Rig stream written: {npz_path} ({npz_size:.0f} KB)")

    calib_path = f"{OUT_DIR}/v13_yaw_calibration.json"
    with open(calib_path, 'w') as f:
        json.dump(calib if calib else {}, f, indent=2)

    elapsed = time.time() - t_start

    mp_mask   = modes_arr == 'MEDIAPIPE'
    rep_mask  = modes_arr == 'REP360'
    hold_mask = modes_arr == 'HOLD'

    print(f"\n{'='*65}")
    print("V15 RIG STREAM SUMMARY")
    print(f"{'='*65}")
    print(f"Frames:       {N}")
    print(f"MEDIAPIPE:    {mp_mask.sum()}  REP360: {rep_mask.sum()}  HOLD: {hold_mask.sum()}")
    print(f"Head anchor: cx={cx_smooth.mean():.0f}±{cx_smooth.std():.0f}  "
          f"cy={cy_smooth.mean():.0f}±{cy_smooth.std():.0f}  "
          f"scale={sc_smooth.mean():.0f}±{sc_smooth.std():.0f}px")
    print(f"Calibration: {'fitted' if calib else 'skipped'}")
    if calib:
        print(f"Calib RMSE: x={calib['rmse_x_px']:.1f}px  y={calib['rmse_y_px']:.1f}px")
    print(f"Output:       {npz_path}")
    print(f"Time:         {elapsed:.0f}s")
    print(f"{'='*65}")

    return npz_path, calib


if __name__ == '__main__':
    run()
