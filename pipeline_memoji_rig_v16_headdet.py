#!/usr/bin/env python3
"""
pipeline_memoji_rig_v16_headdet.py — v16 H1: YOLOv8n-head detector tier (back-of-head coverage).

BUILT ON: pipeline_memoji_rig_v15.py (all v15 changes retained)

H1 CHANGE — YOLO-HEAD FALLBACK TIER (Proposal 3 from RESEARCH_INNOVATE_TRACKING.md):
  Detection hierarchy:
    Tier 1: MediaPipe FaceLandmarker  → ear-midpoint anchor + blendshapes + pose
    Tier 2: YOLOv10n-face → 6DRepNet360 → pose anchor (same as v15)
    Tier 3 (NEW): YOLOv8n-head (SCUT-HEAD trained) → head-box CENTER as position anchor
             POSITION ONLY: 6DRepNet360 cannot estimate pose from a back-of-head crop.
             Orientation is held/extrapolated from the last valid REP360 or MEDIAPIPE frame.
             mode = 'HEAD_DET' in the NPZ.
    Tier 4: Pose-anchor (MediaPipe body ear-midpoint) — unchanged from v15
    Tier 5: HOLD (last position held) — only fires if all 4 tiers above fail

  HEAD_DET Kalman noise:
    R_head_det = 60² = 3600  (head-box center: less precise than face-ear-midpoint ~15px,
                               more trusted than pose_raw ~80px; head box at ~130px span
                               → center error ~1/4 box width ≈ 30-40px, budget = 60px)

  Scale during HEAD_DET frames:
    6DRepNet360 does not run (no face visible for a meaningful crop).
    Scale is inherited from the last valid face/pose measurement and held constant
    through the HEAD_DET region — same as what the Kalman would predict.

v15 changes retained (NOT modified):
  CHANGE 1 — Ear-midpoint anchor from MediaPipe 478-pt mesh (landmarks 234+454)
  CHANGE 2 — Source-aware heteroscedastic Kalman R per anchor source
  FIX 1 (v14) — Jump-gate 150px
  FIX 2 (v14) — RTS vs forward-only fallback
  FIX 3 (v14) — Scale discontinuity segment split

Outputs:
  memoji_rig_stream_v16.npz   (new file — does NOT overwrite v13/v15)
  v16_yaw_calibration.json
  pipeline_version = 'v16'

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
YOLO_FACE_PATH   = "models/yolov10n-face.pt"
# H1: new head detector
YOLO_HEAD_PATH   = "models/yolov8n-head-scut.pt"
REP360_WEIGHTS   = "models/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
OUT_DIR          = "."

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"[v16-rig] Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants (all v15 values retained)
# ─────────────────────────────────────────────────────────────────────────────
BSHP_DECAY   = 0.92
BSHP_NEUTRAL = 0.0
YAW_SIGN     = -1.0
PITCH_SIGN   = -1.0
FOCAL_LEN    = 700.0
VIS_THRESH   = 0.30
MIN_EAR_SPAN = 10.0

# v14 fixes
JUMP_GATE_PX        = 150.0
RTS_MAX_DEV_PX      = 100.0
SCALE_JUMP_RESET_PX = 80.0

# v15 CHANGE 1: MediaPipe 478-pt mesh ear-tragion landmark indices
MP_FACE_EAR_LEFT_IDX  = 234
MP_FACE_EAR_RIGHT_IDX = 454

# v15 CHANGE 2: per-source observation noise R (per-axis variance = sigma²)
R_FACE_EAR    = 15.0 ** 2    # 225
R_POSE_CALIB  = 45.0 ** 2    # 2025
R_POSE_RAW    = 80.0 ** 2    # 6400
R_HOLD        = 500.0 ** 2   # 250000

# H1 (v16): head-box center noise — between face-ear and pose_raw
# SCUT-HEAD box center at ~130px head span → error budget ~60px per axis
R_HEAD_DET    = 60.0 ** 2    # 3600

# H1: head detector conf threshold — keep low to catch profile/back-of-head
HEAD_DET_CONF = 0.20

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
# 6DRepNet360 (unchanged from v15)
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
# MediaPipe landmarkers (unchanged from v15)
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

    return (cx_px, cy_px)


# ─────────────────────────────────────────────────────────────────────────────
# Pose-based head anchor (unchanged from v15)
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
    """
    H1 (v16): Run YOLOv8n-head on the frame; return the best head detection
    center as a position anchor.

    Returns dict with:
      cx, cy  — head box center in pixels
      scale   — head box height in pixels (used as head size proxy for Kalman scale)
      conf    — detection confidence
      box     — [x1, y1, x2, y2] of the winning box

    Returns None if no head detected above HEAD_DET_CONF threshold.

    Design notes:
    - We do NOT run 6DRepNet360 on the head crop for back-of-head frames.
      The head box is occluded from behind; the crop contains hair/scalp, not
      a face — 6DRepNet360 would produce arbitrary pose values.
    - Orientation is held from the last valid REP360/MEDIAPIPE frame by
      state.update_hold() (unchanged) — only POSITION is updated here.
    - Scale is the head-box height, which is a reasonable proxy at back-of-head
      where the ear-span collapses to zero (profile) or is not visible.
    """
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
    # Head box height as scale proxy (more stable than width at profile/back)
    scale = max(y2 - y1, 10.0)

    return {
        'cx': cx,
        'cy': cy,
        'scale': scale,
        'conf': conf,
        'box': [x1, y1, x2, y2],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Canonical mesh utilities (unchanged from v15)
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
# Head state (unchanged from v15 — update_hold() still handles orientation)
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
        """Hold last pose + decay blendshapes. Used for HEAD_DET and HOLD modes."""
        self.blendshapes = self.blendshapes * BSHP_DECAY
        self.frames_since_mp += 1


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — forward inference with H1 head-detector tier
# ─────────────────────────────────────────────────────────────────────────────
def forward_pass(cap: cv2.VideoCapture, fw: int, fh: int,
                 face_lmk, pose_lmk,
                 yolo_face: YOLO, yolo_head: YOLO,
                 rep360: SixDRepNet360,
                 state: HeadState, total_f: int) -> List[Dict]:
    """
    Detection hierarchy per frame:
      1. MediaPipe FaceLandmarker (MEDIAPIPE) — ear-midpoint anchor
      2. YOLOv10n-face → 6DRepNet360 (REP360) — face+pose
      3. YOLOv8n-head (HEAD_DET) — head-box center, orientation held
      4. Pose anchor via PoseLandmarker — calibrated pose (in compute_raw_anchors)
      5. HOLD — last position held

    H1: tier 3 fires when tier 1+2 both fail AND yolo_head detects a head above
    HEAD_DET_CONF=0.20.  Only position anchor (cx, cy, scale) is extracted;
    orientation uses state.update_hold() so the last valid pose is held.
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
        head_anchor = None   # H1: populated if HEAD_DET fires
        face_cx     = None
        face_cy     = None

        if has_mp:
            # Tier 1: MediaPipe face
            state.update_mp(T_mp, B_mp, L_mp, (fh, fw))
            mode = 'MEDIAPIPE'; n_mp += 1

            ear_mid = face_ear_midpoint_anchor(L_mp, fw, fh)
            if ear_mid is not None:
                face_cx, face_cy = ear_mid
                n_ear_ok += 1
            else:
                if pa is not None:
                    face_cx, face_cy = pa['cx'], pa['cy']
                else:
                    face_cx, face_cy = fw / 2.0, fh / 2.0
                n_ear_fallback += 1

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
                    # REP360 failed on face crop — fall through to head detector
                    yolo_box = None
                    ha = head_det_anchor(yolo_head, frame_bgr, fw, fh)
                    if ha is not None:
                        head_anchor = ha
                        state.update_hold()
                        mode = 'HEAD_DET'; n_headdet += 1
                    else:
                        state.update_hold(); mode = 'HOLD'; n_hold += 1
            else:
                # Tier 3: H1 — YOLO-head fallback
                ha = head_det_anchor(yolo_head, frame_bgr, fw, fh)
                if ha is not None:
                    head_anchor = ha
                    state.update_hold()   # orientation held; only position from head box
                    mode = 'HEAD_DET'; n_headdet += 1
                else:
                    state.update_hold(); mode = 'HOLD'; n_hold += 1

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
            'head_anchor':    head_anchor,   # H1: new field
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
# PASS 2A — Yaw-conditioned calibration (unchanged from v15)
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
# PASS 2B — Per-frame raw anchors: v15 logic + H1 head_det tier
# ─────────────────────────────────────────────────────────────────────────────
def compute_raw_anchors(records: List[Dict],
                        calib: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Anchor source priority (per frame):
      1. mediapipe_face  — ear-midpoint (v15 CHANGE 1)
      2. pose_calib/raw  — calibrated/raw pose anchor (v15 fallback)
      3. head_det        — H1: head-box center from YOLOv8n-head (NEW in v16)
      4. predicted       — Kalman predicts (no measurement)

    Note: for HEAD_DET frames, pose_anchor may also be available (body pose).
    We prefer head_det over pose_anchor for position because the head box center
    is directly on the head, whereas body pose ear-midpoint may be unreliable
    when the ears are not visible from behind.  However, if head_det AND
    pose_anchor are both available, we use head_det as primary but set confidence
    from the detection score.
    """
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
        ha = r.get('head_anchor')   # H1: may be None if HEAD_DET did not fire

        if r['mode'] == 'MEDIAPIPE' and r['face_cx'] is not None:
            face_cx = r['face_cx']
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
                cx_raw[i] = face_cx
                cy_raw[i] = face_cy
                sc_raw[i] = pa['scale'] if pa is not None else 80.0
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
            # H1 (v16): use head-box center as position anchor
            cx_raw[i] = ha['cx']
            cy_raw[i] = ha['cy']
            sc_raw[i] = ha['scale']
            sources[i] = 'head_det'
            confs[i]   = float(ha['conf'])
            prev_cx = ha['cx']
            prev_cy = ha['cy']

        elif pa is not None:
            # Pose anchor + calibration (REP360 or HOLD with pose available)
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
# FIX 3 (v14): Scale discontinuity boundaries (unchanged)
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
# v15 CHANGE 2 + H1: per-source R lookup including head_det
# ─────────────────────────────────────────────────────────────────────────────
def r_for_source(source: str) -> float:
    """
    Per-axis observation noise variance R by anchor_source.

      mediapipe_face  → 225   (ear-midpoint, ~15px error)
      head_det        → 3600  (head-box center, ~60px error)  [H1 new]
      pose_calib      → 2025  (~45px RMSE)
      pose_raw        → 6400  (~80px, uncalibrated)
      predicted/hold  → 250000 (no observation)
    """
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
# PASS 2C — Forward-backward Kalman smoother (unchanged from v15, R_HEAD_DET added)
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run():
    t_start = time.time()
    print("[v16-rig] Loading models...")
    face_lmk  = make_mp_landmarker()
    pose_lmk  = make_pose_landmarker()
    yolo_face = YOLO(YOLO_FACE_PATH)
    yolo_head = YOLO(YOLO_HEAD_PATH)   # H1: head detector

    # Verify head detector
    print(f"[v16-rig] Head detector: {YOLO_HEAD_PATH}")
    print(f"[v16-rig] Head detector classes: {yolo_head.names}")
    print(f"[v16-rig] H1 conf threshold: {HEAD_DET_CONF}")
    print(f"[v16-rig] H1 Kalman noise R_head_det={R_HEAD_DET:.0f} (sigma={R_HEAD_DET**0.5:.0f}px)")

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
    print(f"[v16-rig] {total_f} frames @ {fps:.1f}fps  ({fw}x{fh})")

    print("\n[v16-rig] PASS 1: forward inference with H1 head-detector tier...")
    records = forward_pass(cap, fw, fh, face_lmk, pose_lmk,
                           yolo_face, yolo_head, rep360, state, total_f)
    cap.release()
    face_lmk.close()
    pose_lmk.close()

    print("\n[v16-rig] PASS 2A: yaw-conditioned calibration...")
    calib = fit_yaw_calibration(records)

    print("[v16-rig] PASS 2B: computing raw anchors (H1 + jump-gate + ear-midpoint)...")
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

    print("[v16-rig] PASS 2C: heteroscedastic Kalman smoother (FIX 2 + FIX 3 + CHANGE 2 + R_HEAD_DET)...")
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

    # v16: write to new file — does NOT overwrite v15's memoji_rig_stream_v13.npz
    npz_path = f"{OUT_DIR}/memoji_rig_stream_v16.npz"
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
        pipeline_version  = np.array(['v16']),
    )
    npz_size = os.path.getsize(npz_path) / 1024
    print(f"\n[v16-rig] Rig stream written: {npz_path} ({npz_size:.0f} KB)")

    calib_path = f"{OUT_DIR}/v16_yaw_calibration.json"
    with open(calib_path, 'w') as f:
        json.dump(calib if calib else {}, f, indent=2)

    elapsed = time.time() - t_start

    mp_mask    = modes_arr == 'MEDIAPIPE'
    rep_mask   = modes_arr == 'REP360'
    hdet_mask  = modes_arr == 'HEAD_DET'
    hold_mask  = modes_arr == 'HOLD'

    print(f"\n{'='*70}")
    print("V16 RIG STREAM SUMMARY (H1 head-detector)")
    print(f"{'='*70}")
    print(f"Frames:       {N}")
    print(f"MEDIAPIPE:    {mp_mask.sum():4d}  ({100*mp_mask.sum()/N:.1f}%)")
    print(f"REP360:       {rep_mask.sum():4d}  ({100*rep_mask.sum()/N:.1f}%)")
    print(f"HEAD_DET:     {hdet_mask.sum():4d}  ({100*hdet_mask.sum()/N:.1f}%)  [H1 new — was HOLD in v15]")
    print(f"HOLD:         {hold_mask.sum():4d}  ({100*hold_mask.sum()/N:.1f}%)  [target ≤5 from 15 in v15]")
    print(f"Head anchor: cx={cx_smooth.mean():.0f}±{cx_smooth.std():.0f}  "
          f"cy={cy_smooth.mean():.0f}±{cy_smooth.std():.0f}  "
          f"scale={sc_smooth.mean():.0f}±{sc_smooth.std():.0f}px")
    if calib:
        print(f"Calib RMSE: x={calib['rmse_x_px']:.1f}px  y={calib['rmse_y_px']:.1f}px")
    print(f"Output:       {npz_path}")
    print(f"Time:         {elapsed:.0f}s")
    print(f"{'='*70}")

    return npz_path, calib, {
        'n_mediapipe': int(mp_mask.sum()),
        'n_rep360':    int(rep_mask.sum()),
        'n_head_det':  int(hdet_mask.sum()),
        'n_hold':      int(hold_mask.sum()),
        'n_total':     N,
    }


if __name__ == '__main__':
    run()
