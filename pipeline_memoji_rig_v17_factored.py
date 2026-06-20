#!/usr/bin/env python3
"""
pipeline_memoji_rig_v17_factored.py — v17: FACTORIZED avatar output stream.

BUILT ON: pipeline_memoji_rig_v16_headdet.py (all v16/v15/v14 changes retained)

V17 CORE CHANGE — FACTORIZED OUTPUT REPRESENTATION:
  The single flat "mode" + "anchor_source" record is replaced with four
  per-dimension sub-records, each carrying its own source + confidence:

    position:    (cx, cy)  | pos_source  | pos_sigma_px
    scale:       value     | scale_source
    orientation: (yaw, pitch, roll) | orient_source | rot_sigma_deg | orient_observed: bool
    expression:  52 blendshapes     | expr_source   | expr_conf

  CRITICAL DESIGN RULE:
    - head_det / body_pose / SAM2 update POSITION + SCALE only.
    - They MUST set orient_observed=False and leave orientation as
      HELD / DECAYED from the last valid REP360 or MEDIAPIPE frame.
    - rot_sigma_deg RISES each frame that orientation is not observed.
    - The value stored in yaw/pitch/roll on HEAD_DET frames is the
      last-valid held value — it must NEVER be presented as a fresh
      measurement. The orient_observed=False flag is the honest signal.

  FAILURE REASON:
    failure_reason is a per-frame string. Empty string = no failure.
    Non-empty = degraded mode explanation (e.g. "back_of_head_no_orient",
    "hold_all_detectors_failed", "mp_jump_rejected").

DETECTORS UNCHANGED FROM v16:
  Tier 1: MediaPipe FaceLandmarker  (ear-midpoint + blendshapes + pose matrix)
  Tier 2: YOLOv10n-face → 6DRepNet360 (REP360)
  Tier 3: YOLOv8n-head (SCUT-HEAD) — HEAD_DET — position only
  Tier 4: MediaPipe PoseLandmarker (body ear-midpoint)
  Tier 5: HOLD (Kalman predicts)
  Jump-gate (150px), RTS smoother, heteroscedastic Kalman, scale-segment split — all unchanged.

ORIENTATION SIGMA DECAY MODEL:
  - MEDIAPIPE frame: rot_sigma_deg = 2.0 (MP facial_transformation_matrix)
  - REP360 frame:    rot_sigma_deg = 5.0 (6DRepNet360 empirical)
  - HEAD_DET/HOLD:   rot_sigma_deg = prev_rot_sigma + ORIENT_SIGMA_RISE_PER_FRAME
    ORIENT_SIGMA_RISE_PER_FRAME = 3.0 deg/frame (so 15 frames → +45 deg uncertainty)
    orient_observed = False on these frames

OUTPUT FILES:
  memoji_rig_stream_v17.npz  — factorized stream
  v17_factored_montage.png   — diagnostic montage with per-dimension source labels
  notes_v17_factored.md      — honest findings report

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
# Paths (all identical to v16)
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH       = "input_clip.mov"
FACE_MODEL_TASK  = "models/face_landmarker.task"
POSE_MODEL_TASK  = "models/pose_landmarker_full.task"
CANONICAL_OBJ    = "assets/canonical_face_model.obj"
YOLO_FACE_PATH   = "models/yolov10n-face.pt"
YOLO_HEAD_PATH   = "models/yolov8n-head-scut.pt"
REP360_WEIGHTS   = "models/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
OUT_DIR          = "."

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"[v17-factored] Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants (all v16 values retained)
# ─────────────────────────────────────────────────────────────────────────────
BSHP_DECAY   = 0.92
BSHP_NEUTRAL = 0.0
YAW_SIGN     = -1.0
PITCH_SIGN   = -1.0
FOCAL_LEN    = 700.0
VIS_THRESH   = 0.30
MIN_EAR_SPAN = 10.0

JUMP_GATE_PX        = 150.0
RTS_MAX_DEV_PX      = 100.0
SCALE_JUMP_RESET_PX = 80.0

MP_FACE_EAR_LEFT_IDX  = 234
MP_FACE_EAR_RIGHT_IDX = 454

# Per-source position Kalman R (unchanged from v16)
R_FACE_EAR    = 15.0 ** 2    # 225
R_POSE_CALIB  = 45.0 ** 2    # 2025
R_POSE_RAW    = 80.0 ** 2    # 6400
R_HOLD        = 500.0 ** 2   # 250000
R_HEAD_DET    = 60.0 ** 2    # 3600
HEAD_DET_CONF = 0.20

# V17: per-source orientation sigma (degrees) — used to populate rot_sigma_deg
ORIENT_SIGMA_MP       = 2.0   # MediaPipe facial_transformation_matrix
ORIENT_SIGMA_REP360   = 5.0   # 6DRepNet360 empirical estimate
ORIENT_SIGMA_RISE_PER_FRAME = 3.0  # deg/frame added when orient is NOT observed

# V17: per-source position sigma (px) — used to populate pos_sigma_px
POS_SIGMA_MP        = 15.0
POS_SIGMA_HEAD_DET  = 60.0
POS_SIGMA_POSE_CALIB = 45.0
POS_SIGMA_POSE_RAW   = 80.0
POS_SIGMA_HOLD       = 500.0

# V17: expression confidence decay
EXPR_CONF_DECAY = 0.92   # same as BSHP_DECAY — mirrors blendshape hold

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
# 6DRepNet360 (unchanged from v16)
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
# MediaPipe landmarkers (unchanged from v16)
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
# v15 CHANGE 1: Ear-midpoint anchor from MediaPipe face landmarks
# ─────────────────────────────────────────────────────────────────────────────
def face_ear_midpoint_anchor(L_mp: np.ndarray, fw: int, fh: int) -> Optional[Tuple[float, float]]:
    if L_mp is None:
        return None
    n_lm = L_mp.shape[0]
    if n_lm <= max(MP_FACE_EAR_LEFT_IDX, MP_FACE_EAR_RIGHT_IDX):
        return None

    l_ear = L_mp[MP_FACE_EAR_LEFT_IDX]
    r_ear = L_mp[MP_FACE_EAR_RIGHT_IDX]

    l_ear_px = (l_ear[0] * fw, l_ear[1] * fh)
    r_ear_px = (r_ear[0] * fw, r_ear[1] * fh)

    cx_px = 0.5 * (l_ear_px[0] + r_ear_px[0])
    cy_px = 0.5 * (l_ear_px[1] + r_ear_px[1])

    margin = 0.1 * max(fw, fh)
    if not (-margin <= cx_px <= fw + margin and -margin <= cy_px <= fh + margin):
        return None

    ear_span = math.sqrt((l_ear_px[0]-r_ear_px[0])**2 + (l_ear_px[1]-r_ear_px[1])**2)
    return (cx_px, cy_px), ear_span


# ─────────────────────────────────────────────────────────────────────────────
# Pose-based head anchor (unchanged from v16)
# ─────────────────────────────────────────────────────────────────────────────
def pose_head_anchor(result, fw: int, fh: int) -> Optional[Dict]:
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
# H1 (v16): Head-box center anchor from YOLOv8n-head
# ─────────────────────────────────────────────────────────────────────────────
def head_det_anchor(yolo_head: YOLO, frame_bgr: np.ndarray,
                    fw: int, fh: int) -> Optional[Dict]:
    results = yolo_head(frame_bgr, verbose=False, conf=HEAD_DET_CONF, device=str(DEVICE))
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = boxes.conf.argmax().item()
    xyxy = boxes.xyxy[best].tolist()
    conf = float(boxes.conf[best].item())
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    scale = max(y2 - y1, 10.0)

    return {
        'cx': cx,
        'cy': cy,
        'scale': scale,
        'conf': conf,
        'box': [x1, y1, x2, y2],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Canonical mesh utilities (unchanged from v16)
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
# Head state (extended for factorized orientation tracking)
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

        # V17: orientation uncertainty state
        # rot_sigma_deg tracks current uncertainty; rises when unobserved
        self.rot_sigma_deg: float = ORIENT_SIGMA_MP   # starts optimistic
        self.orient_observed: bool = False
        self.orient_source: str = 'none'
        # frames since last valid orientation measurement
        self.frames_since_orient: int = 0

        # V17: expression confidence — starts at 0, set to 1.0 on MP frames
        self.expr_conf: float = 0.0
        self.expr_source: str = 'none'

    def update_mp(self, T: np.ndarray, bshps: np.ndarray, *_):
        """MediaPipe: updates ALL dimensions."""
        self.head_transform = T.copy()
        self.blendshapes    = bshps.copy()
        self.frames_since_mp = 0
        # V17: orient is observed with low sigma
        self.rot_sigma_deg   = ORIENT_SIGMA_MP
        self.orient_observed = True
        self.orient_source   = 'mediapipe'
        self.frames_since_orient = 0
        # Expression: fresh from MediaPipe
        self.expr_conf   = 1.0
        self.expr_source = 'mediapipe'

    def update_rep360(self, R: np.ndarray, box_xyxy: List[float],
                      frame_shape: Tuple[int, int]):
        """REP360: updates orientation + position (via transform); no expression."""
        self.last_yolo_box = box_xyxy
        T = build_head_transform_from_R(R, box_xyxy, frame_shape, self.canon_scale)
        self.head_transform = T.copy()
        self.blendshapes = self.blendshapes * BSHP_DECAY
        self.frames_since_mp += 1
        # V17: orient is observed but with higher sigma than MP
        self.rot_sigma_deg   = ORIENT_SIGMA_REP360
        self.orient_observed = True
        self.orient_source   = 'rep360'
        self.frames_since_orient = 0
        # Expression: decaying hold
        self.expr_conf   = max(self.expr_conf * EXPR_CONF_DECAY, 0.0)
        self.expr_source = 'hold_decay'

    def update_position_only(self, failure_category: str):
        """
        HEAD_DET or HOLD: updates NOTHING except blendshape decay.
        Orientation is NOT observed — only position is updated externally.
        rot_sigma_deg RISES.
        orient_observed = False.

        The transform matrix (head_transform) is NOT touched here — it retains
        the last valid pose. This is the "hold" behavior for the avatar rig.
        The factorized output will expose this as orient_observed=False + rising sigma.
        """
        self.blendshapes = self.blendshapes * BSHP_DECAY
        self.frames_since_mp += 1
        # V17: orientation NOT observed — sigma rises
        self.rot_sigma_deg   = min(
            self.rot_sigma_deg + ORIENT_SIGMA_RISE_PER_FRAME,
            180.0   # cap at 180 deg (maximum meaningful uncertainty)
        )
        self.orient_observed = False
        self.orient_source   = f'held_{failure_category}'
        self.frames_since_orient += 1
        # Expression: decaying
        self.expr_conf   = max(self.expr_conf * EXPR_CONF_DECAY, 0.0)
        self.expr_source = 'hold_decay'

    # legacy alias used in forward_pass for HOLD tier
    def update_hold(self):
        self.update_position_only('hold')


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — forward inference (identical cascade to v16, extended records)
# ─────────────────────────────────────────────────────────────────────────────
def forward_pass(cap: cv2.VideoCapture, fw: int, fh: int,
                 face_lmk, pose_lmk,
                 yolo_face: YOLO, yolo_head: YOLO,
                 rep360: SixDRepNet360,
                 state: HeadState, total_f: int) -> List[Dict]:
    """
    Detection hierarchy: same as v16 — no change to detectors.
    Extended records include per-dimension factorized fields.
    """
    records = []
    n_mp = 0; n_rep360 = 0; n_headdet = 0; n_hold = 0
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

        yolo_box    = None
        head_anchor = None
        face_cx     = None
        face_cy     = None
        mp_ear_span = None
        failure_reason = ''

        if has_mp:
            # Tier 1: MediaPipe face — updates ALL dimensions
            state.update_mp(T_mp, B_mp, L_mp, (fh, fw))
            mode = 'MEDIAPIPE'; n_mp += 1

            ear_result = face_ear_midpoint_anchor(L_mp, fw, fh)
            if ear_result is not None:
                (face_cx, face_cy), mp_ear_span = ear_result
                n_ear_ok += 1
            else:
                if pa is not None:
                    face_cx, face_cy = pa['cx'], pa['cy']
                else:
                    face_cx, face_cy = fw / 2.0, fh / 2.0
                n_ear_fallback += 1
                failure_reason = 'mp_ear_midpoint_fallback'

        else:
            # Tier 2: YOLO-face → 6DRepNet360
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
                    yolo_box = None
                    ha = head_det_anchor(yolo_head, frame_bgr, fw, fh)
                    if ha is not None:
                        head_anchor = ha
                        state.update_position_only('head_det')
                        mode = 'HEAD_DET'; n_headdet += 1
                        failure_reason = 'back_of_head_no_orient'
                    else:
                        state.update_hold(); mode = 'HOLD'; n_hold += 1
                        failure_reason = 'hold_all_detectors_failed'
            else:
                # Tier 3: YOLO-head fallback
                ha = head_det_anchor(yolo_head, frame_bgr, fw, fh)
                if ha is not None:
                    head_anchor = ha
                    state.update_position_only('head_det')
                    mode = 'HEAD_DET'; n_headdet += 1
                    failure_reason = 'back_of_head_no_orient'
                else:
                    state.update_hold(); mode = 'HOLD'; n_hold += 1
                    failure_reason = 'hold_all_detectors_failed'

        T = state.head_transform
        R3 = T[:3, :3]
        col_norm = np.linalg.norm(R3[:, 0])
        if col_norm > 1e-6:
            R3_unit = R3 / col_norm
        else:
            R3_unit = R3.copy()
        euler = Rotation.from_matrix(R3_unit).as_euler('YXZ', degrees=True)
        yaw, pitch, roll = euler[0], euler[1], euler[2]

        # V17: position scale for factorized output
        # For MEDIAPIPE frames, scale = ear_span from landmarks (if available)
        # For REP360 frames, scale = from the yolo box built into transform
        # For HEAD_DET frames, scale = bbox height from head_anchor
        # For HOLD frames, scale = last known
        if mp_ear_span is not None:
            mp_scale = mp_ear_span
        elif pa is not None and pa.get('scale'):
            mp_scale = pa['scale']
        else:
            mp_scale = None

        # Determine pos_source and pos_sigma for this frame
        if mode == 'MEDIAPIPE':
            if mp_ear_span is not None and mp_ear_span >= MIN_EAR_SPAN:
                pos_source   = 'mp_ear_midpoint'
                pos_sigma_px = POS_SIGMA_MP
                scale_source = 'mp_ear_span'
            else:
                pos_source   = 'mp_fallback_pose'
                pos_sigma_px = POS_SIGMA_POSE_CALIB
                scale_source = 'pose_body'
        elif mode == 'REP360':
            pos_source   = 'rep360_calib'
            pos_sigma_px = POS_SIGMA_POSE_CALIB
            scale_source = 'yolo_face_box'
        elif mode == 'HEAD_DET':
            pos_source   = 'head_det'
            pos_sigma_px = POS_SIGMA_HEAD_DET
            scale_source = 'head_det_bbox_height'
        else:  # HOLD
            pos_source   = 'hold_predicted'
            pos_sigma_px = POS_SIGMA_HOLD
            scale_source = 'hold_predicted'

        rec = {
            'frame':         fidx,
            'mode':          mode,
            'head_transform': T.copy(),
            'blendshapes':    state.blendshapes.copy(),
            # raw orientation values (may be HELD on HEAD_DET/HOLD frames)
            'yaw_deg':        float(yaw),
            'pitch_deg':      float(pitch),
            'roll_deg':       float(roll),
            'pose_anchor':    pa,
            'face_cx':        face_cx,
            'face_cy':        face_cy,
            'mp_scale':       mp_scale,
            'yolo_box':       yolo_box,
            'head_anchor':    head_anchor,
            'failure_reason': failure_reason,
            # V17 factorized dimension fields
            'pos_source':     pos_source,
            'pos_sigma_px':   pos_sigma_px,
            'scale_source':   scale_source,
            'orient_observed': state.orient_observed,
            'orient_source':  state.orient_source,
            'rot_sigma_deg':  state.rot_sigma_deg,
            'expr_conf':      float(state.expr_conf),
            'expr_source':    state.expr_source,
            'frames_since_orient': state.frames_since_orient,
        }
        records.append(rec)

        if fidx % 100 == 0:
            e = time.time() - t0
            fps_p = (fidx+1) / max(e, 0.01)
            print(f"  [fwd] f{fidx}/{total_f}: MP={n_mp} REP360={n_rep360} "
                  f"HEAD_DET={n_headdet} HOLD={n_hold} "
                  f"ear_ok={n_ear_ok}  {fps_p:.1f}fps")

    print(f"  [fwd] Done: MP={n_mp} REP360={n_rep360} HEAD_DET={n_headdet} HOLD={n_hold} "
          f"ear_ok={n_ear_ok} ear_fallback={n_ear_fallback}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2A — Yaw-conditioned calibration (unchanged from v16)
# ─────────────────────────────────────────────────────────────────────────────
def fit_yaw_calibration(records: List[Dict]) -> Optional[Dict]:
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
# PASS 2B — Per-frame raw anchors (same logic as v16 + factorized sigma)
# ─────────────────────────────────────────────────────────────────────────────
def compute_raw_anchors(records: List[Dict],
                        calib: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                        List[str], np.ndarray]:
    N = len(records)
    cx_raw  = np.full(N, np.nan, dtype=np.float64)
    cy_raw  = np.full(N, np.nan, dtype=np.float64)
    sc_raw  = np.full(N, np.nan, dtype=np.float64)
    sources = ['unknown'] * N
    confs   = np.zeros(N, dtype=np.float64)

    prev_cx: Optional[float] = None
    prev_cy: Optional[float] = None
    n_jump_rejected = 0

    for i, r in enumerate(records):
        pa = r['pose_anchor']
        ha = r.get('head_anchor')

        if r['mode'] == 'MEDIAPIPE' and r['face_cx'] is not None:
            face_cx = r['face_cx']
            face_cy = r['face_cy']

            jump_ok = True
            if prev_cx is not None:
                dist = math.sqrt((face_cx - prev_cx)**2 + (face_cy - prev_cy)**2)
                if dist > JUMP_GATE_PX:
                    jump_ok = False
                    n_jump_rejected += 1
                    print(f"  [FIX1-jumpgate] f{i}: MP ear-mid jump {dist:.1f}px > {JUMP_GATE_PX}px — REJECTED")
                    # Update failure_reason in record
                    records[i]['failure_reason'] = 'mp_jump_rejected'

            if jump_ok:
                cx_raw[i] = face_cx
                cy_raw[i] = face_cy
                sc_raw[i] = pa['scale'] if pa is not None else (r['mp_scale'] or 80.0)
                sources[i] = 'mediapipe_face'
                confs[i]   = 1.0
                prev_cx = face_cx
                prev_cy = face_cy
            else:
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
                    sources[i] = 'predicted_jumpgate'
                    confs[i]   = 0.0

        elif r['mode'] == 'HEAD_DET' and ha is not None:
            cx_raw[i] = ha['cx']
            cy_raw[i] = ha['cy']
            sc_raw[i] = ha['scale']
            sources[i] = 'head_det'
            confs[i]   = float(ha['conf'])
            prev_cx = ha['cx']
            prev_cy = ha['cy']

        elif pa is not None:
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
            sources[i] = 'predicted'
            confs[i]   = 0.0

    print(f"  [FIX1-jumpgate] Total jump-rejected frames: {n_jump_rejected}")
    return cx_raw, cy_raw, sc_raw, sources, confs


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 (v14): Scale discontinuity boundaries (unchanged from v16)
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
                      f"{sc_raw[i_prev]:.1f}→{sc_raw[i_curr]:.1f}px ({jump:.1f}px jump)")

    return boundaries


# ─────────────────────────────────────────────────────────────────────────────
# v15 CHANGE 2 + H1: per-source R lookup (unchanged from v16)
# ─────────────────────────────────────────────────────────────────────────────
def r_for_source(source: str) -> float:
    src = str(source)
    if src.startswith('mediapipe_face'):
        return R_FACE_EAR
    elif src == 'head_det':
        return R_HEAD_DET
    elif src.startswith('pose_calib'):
        return R_POSE_CALIB
    elif src.startswith('pose_raw'):
        return R_POSE_RAW
    else:
        return R_HOLD


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2C — Forward-backward Kalman smoother (unchanged from v16)
# ─────────────────────────────────────────────────────────────────────────────
def fb_kalman_smooth(cx_raw: np.ndarray, cy_raw: np.ndarray,
                     sc_raw: np.ndarray,
                     confs: np.ndarray,
                     sources: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(cx_raw)

    dt   = 1.0
    q_px = 4.0

    def r_frame(i: int) -> float:
        base_r = r_for_source(sources[i] if sources else 'predicted')
        if sources and str(sources[i]).startswith('pose_') and confs[i] > 0.01:
            conf_inflate = 1.0 / max(confs[i], 0.3)
            return base_r * conf_inflate
        return base_r

    def run_kalman_segment(meas: np.ndarray, seg_sources: List[str],
                           seg_confs: np.ndarray,
                           start: int, end: int):
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

        first_valid = None
        for j in range(seg_len):
            gi = start + j
            if not np.isnan(meas[gi]) and seg_confs[gi] > 0.0:
                first_valid = j; break

        x = np.zeros(2, dtype=np.float64)
        P = np.eye(2) * 1e4
        if first_valid is not None:
            gi = start + first_valid
            x[0] = meas[gi]; x[1] = 0.0
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
            R_t = r_frame(gi)
            if not np.isnan(m) and c >= 0.0 and R_t < R_HOLD:
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

    boundaries = find_scale_discontinuity_boundaries(sc_raw)
    seg_starts = [0] + boundaries
    seg_ends   = boundaries + [N]
    segments   = list(zip(seg_starts, seg_ends))
    print(f"  [FIX3] Smoother segments: {segments}")

    seg_sources = sources if sources else ['predicted'] * N
    seg_confs   = confs

    def smooth_channel(meas, max_pull_px=40.0):
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

        result = rts_pos.copy()

        for i in range(N):
            if not np.isnan(meas[i]) and confs[i] > 0.0:
                raw = meas[i]
                delta = result[i] - raw
                if abs(delta) > max_pull_px:
                    result[i] = raw + np.sign(delta) * max_pull_px

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
# Diagnostic montage (v17 factorized visualization)
# ─────────────────────────────────────────────────────────────────────────────
def build_montage(records: List[Dict],
                  cx_smooth: np.ndarray,
                  cy_smooth: np.ndarray,
                  sc_smooth: np.ndarray,
                  out_path: str,
                  fw: int, fh: int):
    """
    Build a 2-panel diagnostic montage:
    Panel A (top): all frames, colored by orient_observed
                   - green  = orient_observed=True (MEDIAPIPE or REP360)
                   - red    = orient_observed=False (HEAD_DET or HOLD)
                   x-axis = frame index, y-axis = rot_sigma_deg
    Panel B (middle): pos_sigma_px colored by pos_source
    Panel C (bottom): expr_conf over time

    Saved as a tall PNG.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [montage] matplotlib not available — skipping montage")
        return

    N = len(records)
    frames_idx  = np.array([r['frame'] for r in records])
    rot_sigma   = np.array([r['rot_sigma_deg'] for r in records])
    orient_obs  = np.array([r['orient_observed'] for r in records], dtype=bool)
    pos_sigma   = np.array([r['pos_sigma_px'] for r in records])
    expr_conf   = np.array([r['expr_conf'] for r in records])
    modes       = np.array([r['mode'] for r in records])
    pos_sources = np.array([r['pos_source'] for r in records])
    failure_r   = np.array([r['failure_reason'] for r in records])

    # HEAD_DET frame indices
    hd_mask = modes == 'HEAD_DET'
    hd_idx  = np.where(hd_mask)[0]

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    fig.suptitle('v17 Factorized Stream — Per-Dimension Source + Confidence', fontsize=13, fontweight='bold')

    # ── Panel 0: orientation sigma ─────────────────────────────────────────
    ax = axes[0]
    # green where observed, red where not
    for i in range(N - 1):
        color = '#2ca02c' if orient_obs[i] else '#d62728'
        ax.fill_between([frames_idx[i], frames_idx[i+1]],
                        [rot_sigma[i], rot_sigma[i+1]], alpha=0.7, color=color)
    ax.plot(frames_idx, rot_sigma, 'k-', linewidth=0.5, alpha=0.5)
    # mark HEAD_DET frames
    for fi in hd_idx:
        ax.axvline(frames_idx[fi], color='orange', linewidth=0.8, alpha=0.6, linestyle='--')
    ax.set_ylabel('rot_sigma_deg', fontsize=9)
    ax.set_title('Orientation uncertainty (green=observed, red=UNOBSERVED/held, orange lines=HEAD_DET frames)', fontsize=9)
    ax.axhline(ORIENT_SIGMA_REP360, color='blue', linewidth=0.7, linestyle=':', label=f'REP360 sigma={ORIENT_SIGMA_REP360}°')
    ax.axhline(ORIENT_SIGMA_MP, color='cyan', linewidth=0.7, linestyle=':', label=f'MP sigma={ORIENT_SIGMA_MP}°')
    ax.set_ylim(0, max(rot_sigma.max() * 1.1, 20))
    ax.legend(fontsize=7, loc='upper right')

    # Annotate HEAD_DET frames with "UNOBSERVED" label
    if len(hd_idx) > 0:
        mid_hd = hd_idx[len(hd_idx)//2]
        ax.annotate('orient_observed=False\n(back-of-head)',
                    xy=(frames_idx[mid_hd], rot_sigma[mid_hd]),
                    xytext=(frames_idx[mid_hd] + 30, rot_sigma[mid_hd] + 5),
                    fontsize=7, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    # ── Panel 1: yaw values, colored by orient_observed ─────────────────────
    ax = axes[1]
    yaw_vals = np.array([r['yaw_deg'] for r in records])
    # color by orient_observed
    obs_mask = orient_obs
    ax.scatter(frames_idx[obs_mask], yaw_vals[obs_mask],
               c='#2ca02c', s=2, label='orient_observed=True', alpha=0.7)
    ax.scatter(frames_idx[~obs_mask], yaw_vals[~obs_mask],
               c='#d62728', s=6, marker='x', label='orient_observed=False (HELD)', alpha=0.9)
    for fi in hd_idx:
        ax.axvline(frames_idx[fi], color='orange', linewidth=0.8, alpha=0.4, linestyle='--')
    ax.set_ylabel('yaw_deg', fontsize=9)
    ax.set_title('Yaw values — red×=HELD (not measured), green=measured', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.axhline(0, color='k', linewidth=0.3)

    # ── Panel 2: position sigma by source ────────────────────────────────────
    ax = axes[2]
    src_colors = {
        'mp_ear_midpoint': '#1f77b4',
        'rep360_calib':    '#ff7f0e',
        'head_det':        '#d62728',
        'hold_predicted':  '#7f7f7f',
        'mp_fallback_pose': '#9467bd',
        'unknown':         '#bcbd22',
    }
    # group by source
    for src_name, col in src_colors.items():
        src_mask = np.array([r['pos_source'] == src_name for r in records])
        if src_mask.any():
            ax.scatter(frames_idx[src_mask], pos_sigma[src_mask],
                       c=col, s=3, label=src_name, alpha=0.8)
    ax.set_ylabel('pos_sigma_px', fontsize=9)
    ax.set_title('Position uncertainty by source', fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    # mark HEAD_DET zone
    for fi in hd_idx:
        ax.axvline(frames_idx[fi], color='orange', linewidth=0.8, alpha=0.4, linestyle='--')

    # ── Panel 3: expression confidence ───────────────────────────────────────
    ax = axes[3]
    ax.plot(frames_idx, expr_conf, color='#17becf', linewidth=1.0)
    ax.fill_between(frames_idx, expr_conf, alpha=0.3, color='#17becf')
    for fi in hd_idx:
        ax.axvline(frames_idx[fi], color='orange', linewidth=0.8, alpha=0.5, linestyle='--')
    ax.set_ylabel('expr_conf', fontsize=9)
    ax.set_title('Expression confidence (1.0=fresh MP, decays when face absent)', fontsize=9)
    ax.set_xlabel('Frame index', fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  [montage] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY: check HEAD_DET frames have orient_observed=False + rising rot_sigma
# ─────────────────────────────────────────────────────────────────────────────
def verify_factorized_stream(records: List[Dict]) -> Dict:
    """
    CRITICAL VERIFICATION:
    1. All HEAD_DET frames must have orient_observed=False
    2. rot_sigma_deg must be RISING monotonically through HEAD_DET zones
    3. pos_source must be 'head_det' on HEAD_DET frames
    4. MEDIAPIPE frames must have orient_observed=True and pos_source='mp_ear_midpoint'

    Returns a verification report dict.
    """
    modes       = [r['mode'] for r in records]
    orient_obs  = [r['orient_observed'] for r in records]
    pos_sources = [r['pos_source'] for r in records]
    rot_sigmas  = [r['rot_sigma_deg'] for r in records]
    yaw_vals    = [r['yaw_deg'] for r in records]
    fail_reas   = [r['failure_reason'] for r in records]

    hd_idx = [i for i, m in enumerate(modes) if m == 'HEAD_DET']
    mp_idx = [i for i, m in enumerate(modes) if m == 'MEDIAPIPE']

    errors = []
    warnings = []

    # Rule 1: HEAD_DET frames must have orient_observed=False
    hd_orient_false = [i for i in hd_idx if not orient_obs[i]]
    hd_orient_true  = [i for i in hd_idx if orient_obs[i]]
    if hd_orient_true:
        errors.append(f"FAIL: {len(hd_orient_true)} HEAD_DET frames have orient_observed=True — must be False")
    else:
        pass  # good

    # Rule 2: pos_source='head_det' on HEAD_DET frames
    hd_pos_correct = [i for i in hd_idx if pos_sources[i] == 'head_det']
    hd_pos_wrong   = [i for i in hd_idx if pos_sources[i] != 'head_det']
    if hd_pos_wrong:
        errors.append(f"FAIL: {len(hd_pos_wrong)} HEAD_DET frames have pos_source != 'head_det'")

    # Rule 3: rot_sigma must rise through HEAD_DET zones (check per contiguous zone)
    # Find contiguous HEAD_DET zones
    zones = []
    if hd_idx:
        zone_start = hd_idx[0]
        zone_prev  = hd_idx[0]
        for fi in hd_idx[1:]:
            if fi > zone_prev + 2:  # gap of >2 frames = new zone
                zones.append(list(range(zone_start, zone_prev + 1)))
                zone_start = fi
            zone_prev = fi
        zones.append(list(range(zone_start, zone_prev + 1)))

    for zone in zones:
        zone_sigmas = [rot_sigmas[i] for i in zone]
        if len(zone_sigmas) >= 2:
            rising = all(zone_sigmas[k+1] > zone_sigmas[k] for k in range(len(zone_sigmas)-1))
            if not rising:
                deltas = [zone_sigmas[k+1] - zone_sigmas[k] for k in range(len(zone_sigmas)-1)]
                warnings.append(f"WARNING: rot_sigma not monotonically rising in HEAD_DET zone {zone[0]}-{zone[-1]}: deltas={deltas}")

    # Rule 4: MEDIAPIPE frames have orient_observed=True
    mp_orient_true  = [i for i in mp_idx if orient_obs[i]]
    mp_orient_false = [i for i in mp_idx if not orient_obs[i]]
    if mp_orient_false:
        errors.append(f"FAIL: {len(mp_orient_false)} MEDIAPIPE frames have orient_observed=False — must be True")

    # Rule 5: yaw values on HEAD_DET frames are held (same as last valid frame)
    # Check they match the last REP360/MEDIAPIPE frame value
    yaw_held_check = []
    for zone in zones:
        if not zone:
            continue
        # look at the frame just before the zone
        before_idx = zone[0] - 1
        if before_idx >= 0:
            yaw_before = yaw_vals[before_idx]
            yaw_in_zone_0 = yaw_vals[zone[0]]
            if abs(yaw_before - yaw_in_zone_0) < 1.0:
                yaw_held_check.append(f"zone f{zone[0]}-f{zone[-1]}: yaw held correctly at {yaw_in_zone_0:.1f}°")
            else:
                warnings.append(f"WARNING: yaw at start of HEAD_DET zone {zone[0]} changed from {yaw_before:.1f}° to {yaw_in_zone_0:.1f}°")

    # Summary stats for HEAD_DET
    hd_sigma_start = [rot_sigmas[z[0]] for z in zones if z]
    hd_sigma_end   = [rot_sigmas[z[-1]] for z in zones if z]

    report = {
        'n_head_det_frames': len(hd_idx),
        'n_mediapipe_frames': len(mp_idx),
        'head_det_frames': hd_idx,
        'hd_orient_observed_false': len(hd_orient_false),
        'hd_orient_observed_true_VIOLATIONS': len(hd_orient_true),
        'hd_pos_source_correct': len(hd_pos_correct),
        'hd_pos_source_wrong_VIOLATIONS': len(hd_pos_wrong),
        'mp_orient_observed_true': len(mp_orient_true),
        'mp_orient_observed_false_VIOLATIONS': len(mp_orient_false),
        'hd_zones': [[z[0], z[-1]] for z in zones],
        'hd_sigma_at_zone_start': hd_sigma_start,
        'hd_sigma_at_zone_end':   hd_sigma_end,
        'yaw_held_confirmations': yaw_held_check,
        'errors': errors,
        'warnings': warnings,
        'pass': len(errors) == 0,
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run():
    t_start = time.time()
    print("[v17-factored] Loading models...")
    face_lmk  = make_mp_landmarker()
    pose_lmk  = make_pose_landmarker()
    yolo_face = YOLO(YOLO_FACE_PATH)
    yolo_head = YOLO(YOLO_HEAD_PATH)

    print(f"[v17-factored] Head detector: {YOLO_HEAD_PATH}")
    print(f"[v17-factored] H1 conf threshold: {HEAD_DET_CONF}")
    print(f"[v17-factored] Orient sigma model: MP={ORIENT_SIGMA_MP}°, REP360={ORIENT_SIGMA_REP360}°, rise={ORIENT_SIGMA_RISE_PER_FRAME}°/frame")

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
    print(f"[v17-factored] {total_f} frames @ {fps:.1f}fps  ({fw}x{fh})")

    print("\n[v17-factored] PASS 1: forward inference (v16 cascade)...")
    records = forward_pass(cap, fw, fh, face_lmk, pose_lmk,
                           yolo_face, yolo_head, rep360, state, total_f)
    cap.release()
    face_lmk.close()
    pose_lmk.close()

    print("\n[v17-factored] PASS 2A: yaw-conditioned calibration...")
    calib = fit_yaw_calibration(records)

    print("[v17-factored] PASS 2B: computing raw anchors...")
    cx_raw, cy_raw, sc_raw, sources, confs = compute_raw_anchors(records, calib)
    n_nan = int(np.isnan(cx_raw).sum())
    src_counts = {}
    for s in sources:
        cat = ('mediapipe_face' if s.startswith('mediapipe_face') else
               'head_det'       if s == 'head_det'                 else
               'pose_calib'     if s.startswith('pose_calib')      else
               'pose_raw'       if s.startswith('pose_raw')        else 'predicted')
        src_counts[cat] = src_counts.get(cat, 0) + 1
    print(f"  anchor_source counts: " + ", ".join(f"{k}={v}" for k, v in src_counts.items()))
    print(f"  NaN anchor frames: {n_nan}")

    print("[v17-factored] PASS 2C: heteroscedastic Kalman smoother...")
    cx_smooth, cy_smooth, sc_smooth = fb_kalman_smooth(cx_raw, cy_raw, sc_raw, confs, sources)

    # ── Assemble NPZ arrays ─────────────────────────────────────────────────
    N = len(records)
    frames_arr         = np.array([r['frame'] for r in records], dtype=np.int32)
    modes_arr          = np.array([r['mode']  for r in records])
    transforms_arr     = np.array([r['head_transform'] for r in records], dtype=np.float32)
    bshp_arr           = np.array([r['blendshapes'] for r in records], dtype=np.float32)
    yaw_arr            = np.array([r['yaw_deg']   for r in records], dtype=np.float32)
    pitch_arr          = np.array([r['pitch_deg'] for r in records], dtype=np.float32)
    roll_arr           = np.array([r['roll_deg']  for r in records], dtype=np.float32)
    head_center_px     = np.stack([cx_smooth, cy_smooth], axis=1).astype(np.float32)
    head_scale_px      = sc_smooth.astype(np.float32)
    anchor_conf        = confs.astype(np.float32)

    # V17: factorized per-dimension fields
    pos_source_arr     = np.array([r['pos_source']     for r in records])
    pos_sigma_arr      = np.array([r['pos_sigma_px']   for r in records], dtype=np.float32)
    scale_source_arr   = np.array([r['scale_source']   for r in records])
    orient_obs_arr     = np.array([r['orient_observed'] for r in records], dtype=bool)
    orient_source_arr  = np.array([r['orient_source']  for r in records])
    rot_sigma_arr      = np.array([r['rot_sigma_deg']  for r in records], dtype=np.float32)
    expr_conf_arr      = np.array([r['expr_conf']      for r in records], dtype=np.float32)
    expr_source_arr    = np.array([r['expr_source']    for r in records])
    failure_arr        = np.array([r['failure_reason'] for r in records])
    frames_since_orient_arr = np.array([r['frames_since_orient'] for r in records], dtype=np.int32)

    npz_path = f"{OUT_DIR}/memoji_rig_stream_v17.npz"
    np.savez_compressed(
        npz_path,
        # ── standard fields (compatible with v16) ──
        frame             = frames_arr,
        mode              = modes_arr,
        head_transform    = transforms_arr,
        blendshapes       = bshp_arr,
        yaw_deg           = yaw_arr,
        pitch_deg         = pitch_arr,
        roll_deg          = roll_arr,
        head_center_px    = head_center_px,
        head_scale_px     = head_scale_px,
        anchor_source     = np.array(sources),   # Kalman-level source (legacy compat)
        anchor_confidence = anchor_conf,
        arkit_names       = ARKIT_NAMES,
        pipeline_version  = np.array(['v17']),
        # ── V17 FACTORIZED fields ──
        # position dimension
        pos_source        = pos_source_arr,
        pos_sigma_px      = pos_sigma_arr,
        # scale dimension
        scale_source      = scale_source_arr,
        # orientation dimension
        orient_observed   = orient_obs_arr,
        orient_source     = orient_source_arr,
        rot_sigma_deg     = rot_sigma_arr,
        # expression dimension
        expr_conf         = expr_conf_arr,
        expr_source       = expr_source_arr,
        frames_since_orient = frames_since_orient_arr,
        # failure info
        failure_reason    = failure_arr,
    )
    npz_size = os.path.getsize(npz_path) / 1024
    print(f"\n[v17-factored] Rig stream written: {npz_path} ({npz_size:.0f} KB)")

    # ── VERIFY ──────────────────────────────────────────────────────────────
    print("\n[v17-factored] VERIFYING factorized stream...")
    verify_report = verify_factorized_stream(records)

    print("\n  === VERIFICATION REPORT ===")
    print(f"  HEAD_DET frames: {verify_report['n_head_det_frames']}")
    print(f"  HEAD_DET orient_observed=False:  {verify_report['hd_orient_observed_false']}  "
          f"{'PASS' if verify_report['hd_orient_observed_false'] == verify_report['n_head_det_frames'] else 'FAIL'}")
    print(f"  HEAD_DET orient_observed=True VIOLATIONS: {verify_report['hd_orient_observed_true_VIOLATIONS']}")
    print(f"  HEAD_DET pos_source='head_det' correct: {verify_report['hd_pos_source_correct']}  "
          f"{'PASS' if verify_report['hd_pos_source_wrong_VIOLATIONS'] == 0 else 'FAIL'}")
    print(f"  MEDIAPIPE orient_observed=True: {verify_report['mp_orient_observed_true']}  "
          f"{'PASS' if verify_report['mp_orient_observed_false_VIOLATIONS'] == 0 else 'FAIL'}")
    print(f"  HEAD_DET zones: {verify_report['hd_zones']}")
    print(f"  rot_sigma at zone starts: {[f'{v:.1f}°' for v in verify_report['hd_sigma_at_zone_start']]}")
    print(f"  rot_sigma at zone ends:   {[f'{v:.1f}°' for v in verify_report['hd_sigma_at_zone_end']]}")
    print(f"  Yaw held confirmations: {verify_report['yaw_held_confirmations']}")
    if verify_report['errors']:
        print(f"  ERRORS:")
        for e in verify_report['errors']:
            print(f"    {e}")
    if verify_report['warnings']:
        print(f"  Warnings:")
        for w in verify_report['warnings']:
            print(f"    {w}")
    print(f"  OVERALL: {'PASS' if verify_report['pass'] else 'FAIL'}")

    # Per-frame dump of HEAD_DET frames for manual inspection
    print("\n  === HEAD_DET FRAME DETAIL (orient dimension) ===")
    hd_frames = verify_report['head_det_frames']
    for fi in hd_frames:
        r = records[fi]
        print(f"  f{fi:04d}: pos_src={r['pos_source']:18s} pos_sigma={r['pos_sigma_px']:.0f}px  "
              f"orient_obs={r['orient_observed']}  orient_src={r['orient_source']:20s}  "
              f"rot_sigma={r['rot_sigma_deg']:.1f}°  yaw={r['yaw_deg']:.1f}°  "
              f"failure={r['failure_reason']}")

    # ── Montage ─────────────────────────────────────────────────────────────
    montage_path = f"{OUT_DIR}/v17_factored_montage.png"
    print(f"\n[v17-factored] Building diagnostic montage...")
    build_montage(records, cx_smooth, cy_smooth, sc_smooth, montage_path, fw, fh)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    mp_mask    = modes_arr == 'MEDIAPIPE'
    rep_mask   = modes_arr == 'REP360'
    hdet_mask  = modes_arr == 'HEAD_DET'
    hold_mask  = modes_arr == 'HOLD'

    print(f"\n{'='*70}")
    print("V17 FACTORIZED RIG STREAM SUMMARY")
    print(f"{'='*70}")
    print(f"Frames:       {N}")
    print(f"MEDIAPIPE:    {mp_mask.sum():4d}  ({100*mp_mask.sum()/N:.1f}%)")
    print(f"REP360:       {rep_mask.sum():4d}  ({100*rep_mask.sum()/N:.1f}%)")
    print(f"HEAD_DET:     {hdet_mask.sum():4d}  ({100*hdet_mask.sum()/N:.1f}%)")
    print(f"HOLD:         {hold_mask.sum():4d}  ({100*hold_mask.sum()/N:.1f}%)")
    print(f"")
    print(f"FACTORIZED DIMENSION SUMMARY:")
    print(f"  orient_observed=True:  {orient_obs_arr.sum()} frames")
    print(f"  orient_observed=False: {(~orient_obs_arr).sum()} frames  (← should equal HEAD_DET+HOLD)")
    print(f"  rot_sigma range: [{rot_sigma_arr.min():.1f}°, {rot_sigma_arr.max():.1f}°]  mean={rot_sigma_arr.mean():.1f}°")
    print(f"  rot_sigma on HEAD_DET frames: {rot_sigma_arr[hdet_mask].tolist()}")
    print(f"  expr_conf range: [{expr_conf_arr.min():.3f}, {expr_conf_arr.max():.3f}]")
    print(f"  pos_sigma range: [{pos_sigma_arr.min():.0f}px, {pos_sigma_arr.max():.0f}px]")
    print(f"")
    print(f"VERIFICATION: {'PASS' if verify_report['pass'] else 'FAIL'}")
    if calib:
        print(f"Calib RMSE: x={calib['rmse_x_px']:.1f}px  y={calib['rmse_y_px']:.1f}px")
    print(f"Output NPZ:   {npz_path}")
    print(f"Montage:      {montage_path}")
    print(f"Time:         {elapsed:.0f}s")
    print(f"{'='*70}")

    return npz_path, calib, verify_report, {
        'n_mediapipe': int(mp_mask.sum()),
        'n_rep360':    int(rep_mask.sum()),
        'n_head_det':  int(hdet_mask.sum()),
        'n_hold':      int(hold_mask.sum()),
        'n_total':     N,
    }


if __name__ == '__main__':
    run()
