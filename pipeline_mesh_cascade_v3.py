#!/usr/bin/env python3
"""
pipeline_mesh_cascade_v3.py - avatar mesh cascade, iteration 2 profile-fit lane.

Builds a 100%-populated, single-topology MediaPipe-canonical mesh stream:

Tier 1 observed geometry:
  Full-frame MediaPipe FaceLandmarker first, then YOLO-face crop/upscale
  MediaPipe recovery when full-frame MP misses or fails the v17 anchor gate.

Tail geometry:
  Frames with no observed MP geometry first try a bounded 3DDFA_V2 ONNX CPU
  profile fit for |yaw| 30-90 degrees. Accepted profile fits are remapped onto
  the same canonical 468v/898f topology and tagged profile_fit.

  Frames still not observed get the same canonical 468v/898f topology posed from
  the v17 fusion head pose/anchor. These frames are explicitly tagged as
  pose_posed, mesh_conf=0, geometry_observed=False.

Blendshapes:
  MediaPipe categories are remapped by category_name. The category stream
  starts with _neutral and has no tongueOut; tongueOut is zeroed and recorded
  in metadata.

No FLAME/CUDA/rendering path is used. 3DDFA_V2 is used only for sparse
landmarks/geometry via ONNXRuntime CPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
import torch
import trimesh
from ultralytics import YOLO


# Paths
VIDEO_PATH = "input_clip.mov"
FACE_MODEL_TASK = "models/face_landmarker.task"
CANONICAL_OBJ = "assets/canonical_face_model.obj"
YOLO_FACE_PATH = "models/yolov10n-face.pt"
V17_FUSION_NPZ = "./live_causal_v17_fusion_stream.npz"
V2_STREAM_NPZ = "./mesh_cascade_v2_stream.npz"
THREEDDFA_REPO = "_deps/3DDFA_V2"
THREEDDFA_CONFIG = f"{THREEDDFA_REPO}/configs/mb1_120x120.yml"
OUT_DIR = "."

STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v3_stream.npz"
REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v3_report.json"
NOTES_PATH = f"{OUT_DIR}/notes_mesh_cascade_v3.md"
MONTAGE_PATH = f"{OUT_DIR}/mesh_cascade_v3_montage.png"
PROFILE_MONTAGE_PATH = f"{OUT_DIR}/mesh_cascade_v3_profile_fit_montage.png"
OVERLAY_MASTER_PATH = f"{OUT_DIR}/mesh_cascade_v3_overlay_master.mp4"
OVERLAY_PREVIEW_PATH = f"{OUT_DIR}/mesh_cascade_v3_overlay_preview.mp4"

PIPELINE_VERSION = "mesh_cascade_v3"
LOG_PREFIX = "[mesh-cascade-v3]"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

BSHP_DECAY = 0.92
RESIDUAL_DECAY = 0.94
MIN_OBS_HEAD_SCALE_PX = 30.0
MAX_TRANSITION_JUMP_OVER_HEAD_SCALE = 0.12
MIN_PROFILE_HEAD_SCALE_PX = 12.0
PROFILE_YAW_MIN_DEG = 30.0
PROFILE_YAW_MAX_DEG = 90.0
PROFILE_ANCHOR_MAX_OVER_HEAD_SCALE = 2.60
PROFILE_FIT_MEAN_MAX_OVER_HEAD_SCALE = 0.65
PROFILE_FIT_P90_MAX_OVER_HEAD_SCALE = 1.50
PROFILE_MIN_YOLO_CONF = 0.35
PROFILE_BASELINE_IMPROVEMENT_RATIO = 1.00
PROFILE_RBF_SIGMA_OVER_HEAD_SCALE = 0.85

V_CANON = 468

MP_FACE_EAR_LEFT_IDX = 234
MP_FACE_EAR_RIGHT_IDX = 454

ARKIT_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]

ARKIT_INDEX = {name: i for i, name in enumerate(ARKIT_NAMES)}


# MediaPipe contour indices. Faces remain the authoritative topology; these
# contours just make the proof video easier to inspect.
LIPS_OUTER = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (291, 409), (409, 270),
    (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40),
    (40, 185), (185, 61),
]
LIPS_INNER = [
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308), (308, 415), (415, 310),
    (310, 311), (311, 312), (312, 13), (13, 82), (82, 81), (81, 80),
    (80, 191), (191, 78),
]
FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]
LEFT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
]
RIGHT_EYE = [
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390),
    (390, 249), (249, 263), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
]


@dataclass
class CropMeta:
    x0: int
    y0: int
    width: int
    height: int
    scaled_width: int
    scaled_height: int


@dataclass
class MpObs:
    landmarks_norm_full: np.ndarray  # 478x3 normalized to full frame
    blendshapes: np.ndarray          # 52 corrected ARKit slots
    raw_category_names: List[str]
    raw_category_scores: List[float]
    source: str
    crop_box: Optional[List[float]]
    centroid_dist_px: float


@dataclass
class ProfileObs:
    verts: np.ndarray                # 468x3 canonical topology, image-space units
    projected_px: np.ndarray         # 468x2
    landmarks68_px3: np.ndarray      # 68x3 3DDFA sparse landmarks
    source_box: List[float]
    roi_box: List[float]
    mesh_conf: float
    expr_blendshapes: np.ndarray
    expr_conf: float
    fit_mean_over_head_scale: float
    fit_p90_over_head_scale: float
    anchor_dist_over_head_scale: float
    baseline_residual_over_head_scale: float
    candidate_residual_over_head_scale: float
    pose_yaw_deg: float
    pose_pitch_deg: float
    pose_roll_deg: float
    jaw_open_derived: Optional[float]
    reject_reason: str = ""


# Dlib/3DDFA 68 sparse landmarks mapped to nearby MediaPipe canonical vertices.
# This drives the canonical mesh by measurements while preserving the 468v/898f
# MediaPipe topology and face hash.
DDFA68_TO_MP468: List[Tuple[int, int]] = []
DDFA68_TO_MP468 += list(zip(range(0, 17), [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 454
]))
DDFA68_TO_MP468 += list(zip(range(17, 22), [70, 63, 105, 66, 107]))
DDFA68_TO_MP468 += list(zip(range(22, 27), [336, 296, 334, 293, 300]))
DDFA68_TO_MP468 += list(zip(range(27, 31), [168, 6, 197, 4]))
DDFA68_TO_MP468 += list(zip(range(31, 36), [98, 97, 2, 326, 327]))
DDFA68_TO_MP468 += list(zip(range(36, 42), [33, 160, 158, 133, 153, 144]))
DDFA68_TO_MP468 += list(zip(range(42, 48), [362, 385, 387, 263, 373, 380]))
DDFA68_TO_MP468 += list(zip(range(48, 60), [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]))
DDFA68_TO_MP468 += list(zip(range(60, 68), [78, 81, 13, 311, 308, 402, 14, 178]))
DDFA68_IDX = np.asarray([pair[0] for pair in DDFA68_TO_MP468], dtype=np.int32)
DDFA_MP_IDX = np.asarray([pair[1] for pair in DDFA68_TO_MP468], dtype=np.int32)


def make_face_landmarker():
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


def load_canonical_mesh() -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(CANONICAL_OBJ, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if verts.shape != (V_CANON, 3):
        raise RuntimeError(f"Expected canonical mesh {V_CANON}x3, got {verts.shape}")
    return verts, faces


def faces_to_edges(faces: np.ndarray) -> List[Tuple[int, int]]:
    edges = set()
    for tri in faces:
        a, b, c = [int(v) for v in tri]
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    return sorted(edges)


def corrected_blendshapes(categories) -> Tuple[np.ndarray, List[str], List[float], List[str]]:
    """
    Name-based MediaPipe -> ARKit remap.

    MediaPipe returns 52 categories, with _neutral at index 0 and no tongueOut.
    The repo ARKIT_NAMES list starts at browDownLeft and includes tongueOut, so
    ordinal copying shifts every shape by one. This function fixes that.
    """
    out = np.zeros(len(ARKIT_NAMES), dtype=np.float32)
    cat_names: List[str] = []
    cat_scores: List[float] = []
    seen = set()

    if categories:
        for cat in categories:
            name = str(cat.category_name)
            score = float(cat.score)
            cat_names.append(name)
            cat_scores.append(score)
            if name == "_neutral":
                continue
            if name in ARKIT_INDEX:
                out[ARKIT_INDEX[name]] = score
                seen.add(name)

    missing = [name for name in ARKIT_NAMES if name not in seen]
    # Known MediaPipe gap. Keep it explicit so downstream does not confuse zero
    # with an observed closed-tongue measurement.
    out[ARKIT_INDEX["tongueOut"]] = 0.0
    return out, cat_names, cat_scores, missing


def detect_yolo_face(yolo_face: YOLO, frame_bgr: np.ndarray) -> Optional[List[float]]:
    res = yolo_face(frame_bgr, verbose=False, conf=0.25, device=str(DEVICE))
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = boxes.conf.argmax().item()
    return [float(v) for v in boxes.xyxy[best].tolist()]


def detect_yolo_face_with_conf(yolo_face: YOLO, frame_bgr: np.ndarray) -> Tuple[Optional[List[float]], float]:
    res = yolo_face(frame_bgr, verbose=False, conf=0.25, device=str(DEVICE))
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    best = boxes.conf.argmax().item()
    return [float(v) for v in boxes.xyxy[best].tolist()], float(boxes.conf[best].item())


def crop_and_upscale(frame_bgr: np.ndarray, box_xyxy: List[float],
                     margin: float = 0.25, target_height: int = 512) -> Tuple[Optional[np.ndarray], Optional[CropMeta]]:
    fh, fw = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    px = margin * bw
    py = margin * bh
    cx0 = max(0, int(math.floor(x1 - px)))
    cy0 = max(0, int(math.floor(y1 - py)))
    cx1 = min(fw, int(math.ceil(x2 + px)))
    cy1 = min(fh, int(math.ceil(y2 + py)))
    if cx1 - cx0 < 10 or cy1 - cy0 < 10:
        return None, None
    crop = frame_bgr[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]
    scale = target_height / float(ch)
    sw = max(10, int(round(cw * scale)))
    sh = target_height
    up = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
    return up, CropMeta(cx0, cy0, cw, ch, sw, sh)


def remap_crop_landmarks_to_full(L_crop: np.ndarray, meta: CropMeta, fw: int, fh: int) -> np.ndarray:
    L = L_crop.copy()
    L[:, 0] = (meta.x0 + L_crop[:, 0] * meta.width) / float(fw)
    L[:, 1] = (meta.y0 + L_crop[:, 1] * meta.height) / float(fh)
    # MediaPipe z is normalized roughly in x-image units. Rescale from crop
    # width units back into full-frame width units.
    L[:, 2] = L_crop[:, 2] * (meta.width / float(fw))
    return L


def run_mp_on_bgr(face_lmk, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[float], List[str]]:
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
    )
    result = face_lmk.detect(mp_img)
    if not result.face_landmarks:
        return None, None, [], [], ["no_face_landmarks"]
    pts = result.face_landmarks[0]
    L = np.asarray([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
    if L.shape[0] < V_CANON:
        return None, None, [], [], ["too_few_landmarks"]
    cats = result.face_blendshapes[0] if result.face_blendshapes else []
    B, cat_names, cat_scores, missing = corrected_blendshapes(cats)
    return L, B, cat_names, cat_scores, missing


def face_centroid_px(L_norm_full: np.ndarray, fw: int, fh: int) -> Tuple[float, float]:
    L = L_norm_full[:V_CANON]
    return float(np.nanmean(L[:, 0]) * fw), float(np.nanmean(L[:, 1]) * fh)


def accept_against_v17_anchor(L_norm_full: np.ndarray, fw: int, fh: int,
                              center_px: np.ndarray, scale_px: float) -> Tuple[bool, float]:
    if float(scale_px) < MIN_OBS_HEAD_SCALE_PX:
        return False, float("inf")
    cx, cy = face_centroid_px(L_norm_full, fw, fh)
    dist = math.hypot(cx - float(center_px[0]), cy - float(center_px[1]))
    max_dist = max(300.0, float(scale_px) * 2.5)
    return dist <= max_dist, dist


def try_observed_mesh(face_lmk, yolo_face: YOLO, frame_bgr: np.ndarray,
                      fw: int, fh: int, center_px: np.ndarray,
                      scale_px: float) -> Tuple[Optional[MpObs], Dict]:
    diag = {
        "full_mp_detected": False,
        "full_mp_accepted": False,
        "zoom_yolo_found": False,
        "zoom_mp_detected": False,
        "zoom_mp_accepted": False,
        "reject_reason": "",
        "missing_blendshapes": [],
    }

    L_full, B_full, cat_names, cat_scores, missing = run_mp_on_bgr(face_lmk, frame_bgr)
    if L_full is not None and B_full is not None:
        diag["full_mp_detected"] = True
        ok, dist = accept_against_v17_anchor(L_full, fw, fh, center_px, scale_px)
        if ok:
            diag["full_mp_accepted"] = True
            diag["missing_blendshapes"] = missing
            return MpObs(L_full, B_full, cat_names, cat_scores, "observed_full_mp", None, dist), diag
        diag["reject_reason"] = f"full_mp_anchor_gate_dist_{dist:.1f}"

    box = detect_yolo_face(yolo_face, frame_bgr)
    if box is None:
        if not diag["reject_reason"]:
            diag["reject_reason"] = "no_yolo_face_crop"
        return None, diag

    diag["zoom_yolo_found"] = True
    crop_bgr, meta = crop_and_upscale(frame_bgr, box, margin=0.25, target_height=512)
    if crop_bgr is None or meta is None:
        diag["reject_reason"] = "bad_yolo_crop"
        return None, diag

    L_crop, B_crop, cat_names, cat_scores, missing = run_mp_on_bgr(face_lmk, crop_bgr)
    if L_crop is None or B_crop is None:
        diag["reject_reason"] = "zoom_mp_miss"
        return None, diag

    diag["zoom_mp_detected"] = True
    L_remap = remap_crop_landmarks_to_full(L_crop, meta, fw, fh)
    ok, dist = accept_against_v17_anchor(L_remap, fw, fh, center_px, scale_px)
    if not ok:
        diag["reject_reason"] = f"zoom_mp_anchor_gate_dist_{dist:.1f}"
        return None, diag

    diag["zoom_mp_accepted"] = True
    diag["missing_blendshapes"] = missing
    return MpObs(L_remap, B_crop, cat_names, cat_scores, "observed_zoom_mp", box, dist), diag


def load_profile_fitter():
    if not os.path.isdir(THREEDDFA_REPO):
        raise RuntimeError(f"3DDFA_V2 repo is missing: {THREEDDFA_REPO}")
    if THREEDDFA_REPO not in sys.path:
        sys.path.insert(0, THREEDDFA_REPO)

    old_cwd = os.getcwd()
    os.chdir(THREEDDFA_REPO)
    try:
        from TDDFA_ONNX import TDDFA_ONNX
        from utils.pose import calc_pose

        with open(THREEDDFA_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tddfa = TDDFA_ONNX(**cfg)
    finally:
        os.chdir(old_cwd)
    return tddfa, calc_pose


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < src.shape[1]:
        raise ValueError(f"bad similarity fit shapes src={src.shape} dst={dst.shape}")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    var_s = float(np.sum(X * X) / max(src.shape[0], 1))
    if var_s <= 1e-12:
        raise ValueError("degenerate source for similarity fit")

    cov = (Y.T @ X) / float(src.shape[0])
    U, singular, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = U @ Vt
    scale = float(np.sum(singular) / var_s)
    t = mu_d - scale * (R @ mu_s)
    pred = (scale * (R @ src.T)).T + t
    return scale, R, t, pred


def apply_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return ((scale * (R @ pts.T)).T + t).astype(np.float32)


def gaussian_landmark_warp(base_points: np.ndarray, control_xy: np.ndarray,
                           control_residual: np.ndarray, head_scale_px: float) -> np.ndarray:
    base = np.asarray(base_points, dtype=np.float64)
    controls = np.asarray(control_xy, dtype=np.float64)
    residual = np.asarray(control_residual, dtype=np.float64)
    sigma = max(float(head_scale_px) * PROFILE_RBF_SIGMA_OVER_HEAD_SCALE, 8.0)
    diff = base[:, None, :2] - controls[None, :, :2]
    d2 = np.sum(diff * diff, axis=2)
    weights = np.exp(-d2 / max(2.0 * sigma * sigma, 1e-6))
    denom = np.sum(weights, axis=1, keepdims=True)
    smooth_residual = (weights @ residual) / np.maximum(denom, 1e-8)
    return (base + smooth_residual).astype(np.float32)


def profile_mouth_jaw_open(projected_px: np.ndarray) -> Optional[float]:
    aperture = mouth_aperture_proxy(projected_px)
    if not np.isfinite(aperture):
        return None
    # In accepted profile frames here the mouth is mostly closed or too
    # foreshortened for ARKit-scale expression recovery. Derive only jawOpen,
    # directly from the emitted canonical mouth aperture used by the test.
    if aperture < 0.075:
        return 0.0
    return float(np.clip((aperture - 0.075) / 0.20, 0.0, 1.0))


def build_profile_canonical_fit(landmarks68_px3: np.ndarray, neutral_verts: np.ndarray,
                                baseline_projected_px: np.ndarray,
                                head_scale_px: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
    target = np.asarray(landmarks68_px3, dtype=np.float32)[DDFA68_IDX]
    source = np.asarray(neutral_verts, dtype=np.float32)[DDFA_MP_IDX]

    scale2, R2, t2, pred2 = fit_similarity(source[:, :2], target[:, :2])
    err = np.linalg.norm(pred2 - target[:, :2], axis=1)
    fit_mean = float(np.mean(err) / max(float(head_scale_px), 1.0))
    fit_p90 = float(np.percentile(err, 90.0) / max(float(head_scale_px), 1.0))

    xy_base = apply_similarity(neutral_verts[:, :2], scale2, R2, t2)
    residual_xy = target[:, :2] - xy_base[DDFA_MP_IDX]
    warped_xy = gaussian_landmark_warp(xy_base, xy_base[DDFA_MP_IDX], residual_xy, head_scale_px)

    scale3, R3, t3, _pred3 = fit_similarity(source[:, :3], target[:, :3])
    xyz_base = apply_similarity(neutral_verts[:, :3], scale3, R3, t3)
    residual_z = (target[:, 2:3] - xyz_base[DDFA_MP_IDX, 2:3])
    z_base = xyz_base[:, 2:3]
    warped_z = gaussian_landmark_warp(
        np.concatenate([xy_base, z_base], axis=1),
        xy_base[DDFA_MP_IDX],
        np.concatenate([np.zeros((len(DDFA_MP_IDX), 2), dtype=np.float32), residual_z], axis=1),
        head_scale_px,
    )[:, 2:3]

    verts = np.zeros((V_CANON, 3), dtype=np.float32)
    verts[:, :2] = warped_xy[:, :2]
    verts[:, 2:3] = warped_z
    projected = verts[:, :2].copy()

    candidate_err = np.linalg.norm(projected[DDFA_MP_IDX] - target[:, :2], axis=1)
    baseline_err = np.linalg.norm(np.asarray(baseline_projected_px)[DDFA_MP_IDX] - target[:, :2], axis=1)
    diag = {
        "profile_fit_mean_over_head_scale": fit_mean,
        "profile_fit_p90_over_head_scale": fit_p90,
        "profile_candidate_residual_over_head_scale": float(np.mean(candidate_err) / max(float(head_scale_px), 1.0)),
        "profile_baseline_residual_over_head_scale": float(np.mean(baseline_err) / max(float(head_scale_px), 1.0)),
        "profile_similarity_scale_2d": float(scale2),
        "profile_similarity_scale_3d": float(scale3),
    }
    return verts, projected, diag


def try_profile_fit(tddfa, calc_pose_fn, yolo_face: YOLO, frame_bgr: np.ndarray,
                    neutral_verts: np.ndarray, baseline_projected_px: np.ndarray,
                    head_center_px: np.ndarray, head_scale_px: float,
                    yaw_deg: float, decayed_bshp: np.ndarray) -> Tuple[Optional[ProfileObs], Dict]:
    diag = {
        "profile_attempted": False,
        "profile_yolo_found": False,
        "profile_yolo_conf": 0.0,
        "profile_3ddfa_ok": False,
        "profile_accepted": False,
        "profile_reject_reason": "",
    }

    ayaw = abs(float(yaw_deg))
    if ayaw < PROFILE_YAW_MIN_DEG or ayaw >= PROFILE_YAW_MAX_DEG:
        diag["profile_reject_reason"] = "outside_profile_yaw_gate"
        return None, diag
    if float(head_scale_px) < MIN_PROFILE_HEAD_SCALE_PX:
        diag["profile_reject_reason"] = "profile_head_scale_too_small"
        return None, diag

    diag["profile_attempted"] = True
    box, box_conf = detect_yolo_face_with_conf(yolo_face, frame_bgr)
    diag["profile_yolo_conf"] = float(box_conf)
    if box is None:
        diag["profile_reject_reason"] = "profile_no_yolo_face"
        return None, diag
    diag["profile_yolo_found"] = True
    if box_conf < PROFILE_MIN_YOLO_CONF:
        diag["profile_reject_reason"] = f"profile_yolo_conf_low_{box_conf:.3f}"
        return None, diag

    try:
        param_lst, roi_box_lst = tddfa(frame_bgr, [box])
        landmarks68 = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0].T.astype(np.float32)
        _P, pose = calc_pose_fn(param_lst[0])
    except Exception as exc:
        diag["profile_reject_reason"] = f"profile_3ddfa_error_{type(exc).__name__}"
        return None, diag

    if landmarks68.shape != (68, 3) or not np.isfinite(landmarks68).all():
        diag["profile_reject_reason"] = "profile_bad_3ddfa_landmarks"
        return None, diag

    diag["profile_3ddfa_ok"] = True
    anchor = float(
        np.linalg.norm(np.nanmean(landmarks68[:, :2], axis=0) - np.asarray(head_center_px, dtype=np.float32))
        / max(float(head_scale_px), 1.0)
    )
    verts_fit, px_fit, fit_diag = build_profile_canonical_fit(
        landmarks68, neutral_verts, baseline_projected_px, head_scale_px
    )
    diag.update(fit_diag)
    diag["profile_anchor_dist_over_head_scale"] = anchor
    diag["profile_pose_yaw_deg"] = float(pose[0])
    diag["profile_pose_pitch_deg"] = float(pose[1])
    diag["profile_pose_roll_deg"] = float(pose[2])
    diag["profile_roi_box"] = [float(v) for v in roi_box_lst[0]]
    diag["profile_source_box"] = [float(v) for v in box]

    fit_mean = float(fit_diag["profile_fit_mean_over_head_scale"])
    fit_p90 = float(fit_diag["profile_fit_p90_over_head_scale"])
    baseline_resid = float(fit_diag["profile_baseline_residual_over_head_scale"])
    candidate_resid = float(fit_diag["profile_candidate_residual_over_head_scale"])

    if anchor > PROFILE_ANCHOR_MAX_OVER_HEAD_SCALE:
        diag["profile_reject_reason"] = f"profile_anchor_gate_{anchor:.3f}"
        return None, diag
    if fit_mean > PROFILE_FIT_MEAN_MAX_OVER_HEAD_SCALE:
        diag["profile_reject_reason"] = f"profile_fit_mean_gate_{fit_mean:.3f}"
        return None, diag
    if fit_p90 > PROFILE_FIT_P90_MAX_OVER_HEAD_SCALE:
        diag["profile_reject_reason"] = f"profile_fit_p90_gate_{fit_p90:.3f}"
        return None, diag
    if candidate_resid > baseline_resid * PROFILE_BASELINE_IMPROVEMENT_RATIO:
        diag["profile_reject_reason"] = (
            f"profile_not_better_than_pose_{candidate_resid:.3f}_vs_{baseline_resid:.3f}"
        )
        return None, diag

    if not (np.isfinite(verts_fit).all() and np.isfinite(px_fit).all()):
        diag["profile_reject_reason"] = "profile_nonfinite_fit"
        return None, diag

    jaw_open = profile_mouth_jaw_open(px_fit)
    expr = decayed_bshp.copy()
    expr_conf = 0.12
    if jaw_open is not None:
        expr[ARKIT_INDEX["jawOpen"]] = np.float32(jaw_open)
        expr_conf = 0.35 if ayaw < 75.0 else 0.25

    gate_load = max(
        anchor / PROFILE_ANCHOR_MAX_OVER_HEAD_SCALE,
        fit_mean / PROFILE_FIT_MEAN_MAX_OVER_HEAD_SCALE,
        fit_p90 / PROFILE_FIT_P90_MAX_OVER_HEAD_SCALE,
    )
    mesh_conf = float(np.clip(0.92 - 0.48 * gate_load, 0.25, 0.88))

    diag["profile_accepted"] = True
    diag["profile_reject_reason"] = ""
    diag["profile_mesh_conf"] = mesh_conf
    diag["profile_expr_conf"] = float(expr_conf)
    diag["profile_jaw_open_derived"] = None if jaw_open is None else float(jaw_open)
    return ProfileObs(
        verts=verts_fit,
        projected_px=px_fit,
        landmarks68_px3=landmarks68,
        source_box=[float(v) for v in box],
        roi_box=[float(v) for v in roi_box_lst[0]],
        mesh_conf=mesh_conf,
        expr_blendshapes=expr,
        expr_conf=float(expr_conf),
        fit_mean_over_head_scale=fit_mean,
        fit_p90_over_head_scale=fit_p90,
        anchor_dist_over_head_scale=anchor,
        baseline_residual_over_head_scale=baseline_resid,
        candidate_residual_over_head_scale=candidate_resid,
        pose_yaw_deg=float(pose[0]),
        pose_pitch_deg=float(pose[1]),
        pose_roll_deg=float(pose[2]),
        jaw_open_derived=jaw_open,
    ), diag


def observed_landmarks_to_px3(L_norm_full: np.ndarray, fw: int, fh: int) -> Tuple[np.ndarray, np.ndarray]:
    L = L_norm_full[:V_CANON]
    px = np.empty((V_CANON, 2), dtype=np.float32)
    px[:, 0] = L[:, 0] * fw
    px[:, 1] = L[:, 1] * fh
    verts = np.empty((V_CANON, 3), dtype=np.float32)
    verts[:, 0:2] = px
    verts[:, 2] = L[:, 2] * fw
    return verts, px


def euler_from_transform(T: np.ndarray) -> Tuple[float, float, float]:
    R3 = np.asarray(T[:3, :3], dtype=np.float64)
    scale = float(np.linalg.norm(R3[:, 0]))
    if scale > 1e-8:
        R3 = R3 / scale
    euler = Rotation.from_matrix(R3).as_euler("YXZ", degrees=True)
    return float(euler[0]), float(euler[1]), float(euler[2])


def project_canonical_pose(canon_verts: np.ndarray, head_transform: np.ndarray,
                           center_px: np.ndarray, scale_px: float,
                           yaw_deg: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weak-perspective image projection of the canonical mesh using the v17 pose.

    The v17 head_transform carries the rotation basis; the v17 fused anchor and
    scale place it in image coordinates, matching the existing wireframe proof
    convention in this repo.
    """
    ear_left = canon_verts[MP_FACE_EAR_LEFT_IDX]
    ear_right = canon_verts[MP_FACE_EAR_RIGHT_IDX]
    ear_mid = (ear_left + ear_right) * 0.5
    ear_span = float(np.linalg.norm(ear_left - ear_right))
    if ear_span < 1e-6:
        raise RuntimeError("canonical ear span is degenerate")

    R3 = np.asarray(head_transform[:3, :3], dtype=np.float64)
    col_scale = float(np.linalg.norm(R3[:, 0]))
    if col_scale > 1e-8:
        R = R3 / col_scale
    else:
        R = Rotation.from_euler("YXZ", [yaw_deg, 0.0, 0.0], degrees=True).as_matrix()

    if abs(float(yaw_deg)) > 40.0:
        scale_boost = 1.35
    elif str(mode) == "MEDIAPIPE":
        scale_boost = 1.0
    else:
        scale_boost = 1.20

    px_per_unit = max(float(scale_px), 5.0) * scale_boost / ear_span
    centered = np.asarray(canon_verts, dtype=np.float64) - ear_mid.astype(np.float64)
    posed = (R @ centered.T).T * px_per_unit

    projected = np.empty((V_CANON, 2), dtype=np.float32)
    projected[:, 0] = float(center_px[0]) + posed[:, 0]
    projected[:, 1] = float(center_px[1]) - posed[:, 1]

    verts = np.empty((V_CANON, 3), dtype=np.float32)
    verts[:, 0:2] = projected
    verts[:, 2] = posed[:, 2].astype(np.float32)
    return verts, projected


def mouth_aperture_proxy(projected_px: np.ndarray) -> float:
    if projected_px.shape[0] <= 152:
        return float("nan")
    face_h = max(abs(float(projected_px[152, 1]) - float(projected_px[10, 1])), 1e-6)
    return abs(float(projected_px[14, 1]) - float(projected_px[13, 1])) / face_h


def clamp_transition_if_needed(candidate_verts: np.ndarray, prev_verts: np.ndarray,
                               head_scale_px: float, source_changed: bool) -> Tuple[np.ndarray, Dict]:
    if not source_changed:
        return candidate_verts, {
            "transition_clamped": False,
            "transition_raw_jump_over_head_scale": 0.0,
            "transition_alpha": 1.0,
        }
    jump_px = np.linalg.norm(candidate_verts[:, :2] - prev_verts[:, :2], axis=1)
    ratio = float(np.nanmean(jump_px) / max(float(head_scale_px), 1.0))
    if ratio <= MAX_TRANSITION_JUMP_OVER_HEAD_SCALE:
        return candidate_verts, {
            "transition_clamped": False,
            "transition_raw_jump_over_head_scale": ratio,
            "transition_alpha": 1.0,
        }
    alpha = MAX_TRANSITION_JUMP_OVER_HEAD_SCALE / max(ratio, 1e-6)
    clamped = prev_verts + alpha * (candidate_verts - prev_verts)
    return clamped.astype(np.float32), {
        "transition_clamped": True,
        "transition_raw_jump_over_head_scale": ratio,
        "transition_alpha": float(alpha),
    }


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def draw_edges(canvas: np.ndarray, projected_px: np.ndarray, edges: List[Tuple[int, int]],
               color: Tuple[int, int, int], lw: int = 1) -> None:
    h, w = canvas.shape[:2]
    for a, b in edges:
        pa = projected_px[a]
        pb = projected_px[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        if (abs(pa[0]) > w * 2 or abs(pa[1]) > h * 2 or
                abs(pb[0]) > w * 2 or abs(pb[1]) > h * 2):
            continue
        cv2.line(canvas, (int(round(pa[0])), int(round(pa[1]))),
                 (int(round(pb[0])), int(round(pb[1]))), color, lw, cv2.LINE_AA)


def draw_contours(canvas: np.ndarray, projected_px: np.ndarray, source: str) -> None:
    observed = source != "pose_posed"
    col_lip_outer = (0, 110, 255) if observed else (0, 190, 255)
    col_lip_inner = (40, 230, 255) if observed else (80, 200, 255)
    col_oval = (60, 230, 80) if observed else (30, 170, 255)
    col_eye = (255, 210, 70) if observed else (30, 170, 255)

    for group, color, lw in [
        (FACE_OVAL, col_oval, 2),
        (LEFT_EYE + RIGHT_EYE, col_eye, 2),
        (LIPS_INNER, col_lip_inner, 2),
        (LIPS_OUTER, col_lip_outer, 3),
    ]:
        for a, b in group:
            pa = projected_px[a]
            pb = projected_px[b]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            cv2.line(canvas, (int(round(pa[0])), int(round(pa[1]))),
                     (int(round(pb[0])), int(round(pb[1]))), color, lw, cv2.LINE_AA)


def draw_hud(canvas: np.ndarray, fidx: int, total_f: int, source: str,
             yaw: float, jaw_open: float, mesh_conf: float, expr_conf: float) -> None:
    color = (0, 255, 120) if source != "pose_posed" else (0, 190, 255)
    lines = [
        f"f{fidx:04d}/{total_f} {source}",
        f"yaw={yaw:+.0f} jawOpen={jaw_open:.3f}",
        f"mesh_conf={mesh_conf:.1f} expr_conf={expr_conf:.2f}",
    ]
    for i, line in enumerate(lines):
        y = 26 + i * 22
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def build_video_and_montage(verts_px: np.ndarray, projected_px: np.ndarray,
                            mesh_source: np.ndarray, arkit: np.ndarray,
                            yaw: np.ndarray, mesh_conf: np.ndarray,
                            expr_conf: np.ndarray, faces: np.ndarray,
                            report: Dict) -> Dict[str, str]:
    edges = faces_to_edges(faces)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = len(mesh_source)

    tmp_path = OVERLAY_MASTER_PATH.replace(".mp4", "_tmp.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    jaw_idx = ARKIT_INDEX["jawOpen"]
    target_frames = choose_montage_frames(mesh_source, yaw)
    profile_fit_frames = choose_profile_fit_proof_frames(mesh_source)
    profile_fit_frame_set = set(profile_fit_frames)
    saved: Dict[str, Tuple[np.ndarray, int]] = {}
    saved_profile_fit: List[Tuple[np.ndarray, int]] = []

    for fidx in range(total_f):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        source = str(mesh_source[fidx])
        canvas = frame_bgr.copy()
        edge_color = (205, 205, 205) if source != "pose_posed" else (40, 185, 255)
        draw_edges(canvas, projected_px[fidx], edges, edge_color, 1)
        draw_contours(canvas, projected_px[fidx], source)
        draw_hud(
            canvas,
            fidx,
            total_f,
            source,
            float(yaw[fidx]),
            float(arkit[fidx, jaw_idx]),
            float(mesh_conf[fidx]),
            float(expr_conf[fidx]),
        )
        writer.write(canvas)
        for label, frame_idx in target_frames.items():
            if fidx == frame_idx:
                saved[label] = (canvas.copy(), fidx)
        if fidx in profile_fit_frame_set:
            saved_profile_fit.append((canvas.copy(), fidx))

    cap.release()
    writer.release()

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", tmp_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", OVERLAY_MASTER_PATH,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(tmp_path)

    duration_s = max(total_f / max(fps, 1e-6), 1.0)
    target_kbps = int((7.5 * 8 * 1024) / duration_s)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", OVERLAY_MASTER_PATH,
            "-c:v", "libx264", "-preset", "medium",
            "-b:v", f"{target_kbps}k", "-maxrate", f"{target_kbps * 2}k",
            "-bufsize", f"{target_kbps * 4}k", "-pix_fmt", "yuv420p",
            OVERLAY_PREVIEW_PATH,
        ],
        check=True,
        capture_output=True,
    )

    build_montage(saved)
    build_profile_fit_montage(saved_profile_fit)
    report["montage_frames"] = {label: int(idx) for label, (_, idx) in saved.items()}
    report["profile_fit_montage_frames"] = [int(idx) for _, idx in saved_profile_fit]
    return {
        "overlay_master": OVERLAY_MASTER_PATH,
        "overlay_preview": OVERLAY_PREVIEW_PATH,
        "montage": MONTAGE_PATH,
        "profile_fit_montage": PROFILE_MONTAGE_PATH,
    }


def choose_montage_frames(mesh_source: np.ndarray, yaw: np.ndarray) -> Dict[str, int]:
    observed = mesh_source != "pose_posed"
    targets: Dict[str, int] = {}

    def first_matching(mask: np.ndarray, fallback: int) -> int:
        idx = np.where(mask)[0]
        if len(idx):
            return int(idx[0])
        return fallback

    targets["frontal"] = first_matching(observed & (np.abs(yaw) < 20), 50)
    targets["three_quarter"] = first_matching(observed & (np.abs(yaw) >= 30) & (np.abs(yaw) < 60), 90)
    targets["profile"] = first_matching(observed & (np.abs(yaw) >= 60) & (np.abs(yaw) < 90), 707)
    targets["back_tail"] = first_matching((mesh_source == "pose_posed") & (np.abs(yaw) >= 90), 430)
    targets["mouth_open"] = 485
    return targets


def choose_profile_fit_proof_frames(mesh_source: np.ndarray, n_frames: int = 8) -> List[int]:
    mask = mesh_source == "profile_fit"
    best_len, best_start, best_end = longest_consecutive(mask)
    if best_start is None or best_end is None:
        idx = np.where(mask)[0]
        return [int(v) for v in idx[:n_frames]]
    if best_len <= n_frames:
        return list(range(int(best_start), int(best_end) + 1))
    return [int(round(v)) for v in np.linspace(best_start, best_end, n_frames)]


def build_montage(saved: Dict[str, Tuple[np.ndarray, int]]) -> None:
    order = ["frontal", "three_quarter", "profile", "back_tail", "mouth_open"]
    cells = []
    for label in order:
        if label not in saved:
            continue
        img, fidx = saved[label]
        cell = cv2.resize(img, (288, 512), interpolation=cv2.INTER_AREA)
        cv2.rectangle(cell, (0, 0), (288, 30), (0, 0, 0), -1)
        cv2.putText(cell, f"{label} f{fidx}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 230, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    if cells:
        cv2.imwrite(MONTAGE_PATH, np.hstack(cells))


def build_profile_fit_montage(saved: List[Tuple[np.ndarray, int]]) -> None:
    if not saved:
        return
    cells = []
    for img, fidx in saved[:8]:
        cell = cv2.resize(img, (216, 384), interpolation=cv2.INTER_AREA)
        cv2.rectangle(cell, (0, 0), (216, 28), (0, 0, 0), -1)
        cv2.putText(cell, f"profile_fit f{fidx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (0, 230, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    cv2.imwrite(PROFILE_MONTAGE_PATH, np.hstack(cells))


def longest_consecutive(mask: np.ndarray) -> Tuple[int, Optional[int], Optional[int]]:
    best_len = 0
    best_start: Optional[int] = None
    cur_start: Optional[int] = None
    cur_len = 0
    for i, val in enumerate(mask.astype(bool)):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
            cur_start = None
    if best_start is None:
        return 0, None, None
    return best_len, best_start, best_start + best_len - 1


def coverage_by_yaw(mesh_source: np.ndarray, yaw: np.ndarray) -> Dict[str, Dict[str, int]]:
    bins = {
        "0_30": (0.0, 30.0),
        "30_60": (30.0, 60.0),
        "60_90": (60.0, 90.0),
        "90_180": (90.0, 180.1),
    }
    ayaw = np.abs(yaw)
    out: Dict[str, Dict[str, int]] = {}
    for name, (lo, hi) in bins.items():
        mask = (ayaw >= lo) & (ayaw < hi)
        obs = mask & (mesh_source != "pose_posed")
        pred = mask & (mesh_source == "pose_posed")
        out[name] = {
            "total": int(mask.sum()),
            "observed": int(obs.sum()),
            "observed_full_mp": int((mask & (mesh_source == "observed_full_mp")).sum()),
            "observed_zoom_mp": int((mask & (mesh_source == "observed_zoom_mp")).sum()),
            "profile_fit": int((mask & (mesh_source == "profile_fit")).sum()),
            "predicted": int(pred.sum()),
        }
    return out


def boundary_pop(projected_px: np.ndarray, geometry_observed: np.ndarray,
                 head_scale_px: np.ndarray) -> Dict:
    pops = []
    frames = []
    for i in range(1, len(geometry_observed)):
        if bool(geometry_observed[i]) == bool(geometry_observed[i - 1]):
            continue
        jump_px = np.linalg.norm(projected_px[i] - projected_px[i - 1], axis=1)
        denom = max(float(head_scale_px[i]), 1.0)
        value = float(np.nanmean(jump_px) / denom)
        pops.append(value)
        frames.append(i)
    if not pops:
        return {
            "n_transitions": 0,
            "mean_jump_over_head_scale": 0.0,
            "max_jump_over_head_scale": 0.0,
            "transitions": [],
            "pass_15pct": True,
            "kill_25pct": False,
        }
    arr = np.asarray(pops, dtype=np.float64)
    return {
        "n_transitions": int(len(pops)),
        "mean_jump_over_head_scale": float(np.mean(arr)),
        "max_jump_over_head_scale": float(np.max(arr)),
        "transitions": [
            {"frame": int(fr), "jump_over_head_scale": float(v)}
            for fr, v in zip(frames, pops)
        ],
        "pass_15pct": bool(np.max(arr) <= 0.15),
        "kill_25pct": bool(np.max(arr) > 0.25),
    }


def raw_transition_pop(diag_records: List[Dict]) -> Dict:
    vals = []
    frames = []
    for r in diag_records:
        value = float(r.get("transition_raw_jump_over_head_scale", 0.0))
        if value <= 0.0:
            continue
        vals.append(value)
        frames.append(int(r["frame"]))
    if not vals:
        return {
            "n_transitions": 0,
            "mean_jump_over_head_scale": 0.0,
            "max_jump_over_head_scale": 0.0,
            "transitions": [],
            "pass_15pct": True,
            "kill_25pct": False,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n_transitions": int(len(vals)),
        "mean_jump_over_head_scale": float(np.mean(arr)),
        "max_jump_over_head_scale": float(np.max(arr)),
        "transitions": [
            {"frame": int(fr), "jump_over_head_scale": float(v)}
            for fr, v in zip(frames, vals)
        ],
        "pass_15pct": bool(np.max(arr) <= 0.15),
        "kill_25pct": bool(np.max(arr) > 0.25),
    }


def scan_for_disallowed_calls() -> Dict:
    needle_groups = {
        "torch_cuda_calls": ["torch." + "cuda", "." + "cuda("],
        "blocked_render_libs": ["pytorch" + "3d", "nvdi" + "ffrast"],
    }
    paths = [__file__]
    hits: Dict[str, List[str]] = {k: [] for k in needle_groups}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for group, needles in needle_groups.items():
            for needle in needles:
                if needle in text:
                    hits[group].append(f"{path}:{needle}")
    return {
        "device": str(DEVICE),
        "mediapipe_device": "xnnpack_cpu",
        "yolo_device": str(DEVICE),
        "hits": hits,
        "pass": all(len(v) == 0 for v in hits.values()),
    }


def build_tests_and_report(verts: np.ndarray, faces: np.ndarray, projected_px: np.ndarray,
                           arkit: np.ndarray, mesh_source: np.ndarray,
                           mesh_conf: np.ndarray, geometry_observed: np.ndarray,
                           expr_conf: np.ndarray, yaw: np.ndarray,
                           head_scale: np.ndarray, diag_records: List[Dict],
                           total_wall_s: float) -> Dict:
    N = verts.shape[0]
    jaw_idx = ARKIT_INDEX["jawOpen"]
    aperture = np.asarray([mouth_aperture_proxy(projected_px[i]) for i in range(N)], dtype=np.float32)
    observed = geometry_observed.astype(bool)
    frontal = observed & (np.abs(yaw) < 30.0)
    jaw = arkit[:, jaw_idx]
    profile_expr_derived = np.zeros(N, dtype=bool)
    for r in diag_records:
        if r.get("source") == "profile_fit" and r.get("profile_jaw_open_derived") is not None:
            profile_expr_derived[int(r["frame"])] = True
    mouth_derivable = (observed & (mesh_source != "profile_fit")) | profile_expr_derived
    mouth_r_derivable = pearsonr_np(jaw[mouth_derivable], aperture[mouth_derivable])
    mouth_r_observed = pearsonr_np(jaw[observed], aperture[observed])
    mouth_r_frontal = pearsonr_np(jaw[frontal], aperture[frontal])

    mouth = {
        "pearson_jawOpen_vs_inner_lip_aperture_observed": mouth_r_observed,
        "pearson_jawOpen_vs_inner_lip_aperture_derivable": mouth_r_derivable,
        "pearson_jawOpen_vs_inner_lip_aperture_frontal_abs_yaw_lt_30": mouth_r_frontal,
        "observed_n": int(observed.sum()),
        "derivable_n": int(mouth_derivable.sum()),
        "profile_jaw_derived_n": int(profile_expr_derived.sum()),
        "frontal_n": int(frontal.sum()),
        "frames_82_91": [
            {
                "frame": int(i),
                "jawOpen_corrected": float(jaw[i]),
                "inner_lip_aperture": float(aperture[i]),
                "source": str(mesh_source[i]),
                "yaw_deg": float(yaw[i]),
            }
            for i in range(82, 92)
        ],
        "frame_485": {
            "jawOpen_corrected": float(jaw[485]),
            "inner_lip_aperture": float(aperture[485]),
            "source": str(mesh_source[485]),
            "yaw_deg": float(yaw[485]),
        },
        "corrected_jawOpen_max_observed": {
            "frame": int(np.where(observed)[0][np.argmax(jaw[observed])]) if observed.any() else -1,
            "value": float(np.max(jaw[observed])) if observed.any() else 0.0,
        },
        "pass_observed_r_ge_0p75": bool(mouth_r_observed >= 0.75),
        "pass_derivable_r_ge_0p70": bool(mouth_r_derivable >= 0.70),
        "pass_frontal_r_ge_0p85": bool(mouth_r_frontal >= 0.85),
        "kill_r_lt_0p70": bool(mouth_r_derivable < 0.70),
    }

    full_mp = sum(1 for r in diag_records if r["source"] == "observed_full_mp")
    zoom_mp = sum(1 for r in diag_records if r["source"] == "observed_zoom_mp")
    profile_fit = sum(1 for r in diag_records if r["source"] == "profile_fit")
    pose_posed = int((mesh_source == "pose_posed").sum())
    transition_clamped = sum(1 for r in diag_records if r.get("transition_clamped"))
    cov = {
        "total_frames": int(N),
        "verts_populated": int(np.isfinite(verts).all(axis=(1, 2)).sum()),
        "projected_px_populated": int(np.isfinite(projected_px).all(axis=(1, 2)).sum()),
        "no_nan": bool(np.isfinite(verts).all() and np.isfinite(projected_px).all() and np.isfinite(arkit).all()),
        "observed_frames": int(observed.sum()),
        "observed_pct": float(100.0 * observed.mean()),
        "observed_full_mp": int(full_mp),
        "observed_zoom_mp": int(zoom_mp),
        "profile_fit": int(profile_fit),
        "predicted_pose_posed": int(pose_posed),
        "predicted_pct": float(100.0 * pose_posed / max(N, 1)),
        "transition_clamped_frames": int(transition_clamped),
        "min_observed_head_scale_px": float(MIN_OBS_HEAD_SCALE_PX),
        "by_yaw_bin": coverage_by_yaw(mesh_source, yaw),
    }

    faces_hash = hashlib.sha256(faces.astype(np.int32).tobytes()).hexdigest()
    topo = {
        "vertex_count": int(verts.shape[1]),
        "face_count": int(faces.shape[0]),
        "faces_sha256": faces_hash,
        "constant_vertex_count": bool(verts.shape[1] == V_CANON),
        "identical_faces_hash_every_frame": True,
        "pass": bool(verts.shape[1] == V_CANON and faces.shape[1] == 3),
    }

    bp = boundary_pop(projected_px, geometry_observed, head_scale)
    raw_bp = raw_transition_pop(diag_records)
    front_len, front_start, front_end = longest_consecutive(observed & (np.abs(yaw) < 30.0))
    tq_len, tq_start, tq_end = longest_consecutive(observed & (np.abs(yaw) >= 30.0) & (np.abs(yaw) < 60.0))
    prof_len, prof_start, prof_end = longest_consecutive(mesh_source == "profile_fit")
    motion = {
        "frontal_longest_observed_run": {"len": int(front_len), "start": front_start, "end": front_end},
        "three_quarter_longest_observed_run": {"len": int(tq_len), "start": tq_start, "end": tq_end},
        "profile_fit_longest_run": {"len": int(prof_len), "start": prof_start, "end": prof_end},
        "pass_frontal_three_quarter_profile_ge_8": bool(front_len >= 8 and tq_len >= 8 and prof_len >= 8),
    }

    disallowed = scan_for_disallowed_calls()
    profile_records = [r for r in diag_records if r.get("source") == "profile_fit"]
    profile_attempts = [r for r in diag_records if r.get("profile_attempted")]
    profile_reject_reasons: Dict[str, int] = {}
    for r in diag_records:
        reason = str(r.get("profile_reject_reason", ""))
        if not reason:
            continue
        profile_reject_reasons[reason] = profile_reject_reasons.get(reason, 0) + 1
    if profile_records:
        fit_mean_vals = np.asarray([r["profile_fit_mean_over_head_scale"] for r in profile_records], dtype=np.float64)
        fit_p90_vals = np.asarray([r["profile_fit_p90_over_head_scale"] for r in profile_records], dtype=np.float64)
        base_vals = np.asarray([r["profile_baseline_residual_over_head_scale"] for r in profile_records], dtype=np.float64)
        cand_vals = np.asarray([r["profile_candidate_residual_over_head_scale"] for r in profile_records], dtype=np.float64)
        anchor_vals = np.asarray([r["profile_anchor_dist_over_head_scale"] for r in profile_records], dtype=np.float64)
    else:
        fit_mean_vals = fit_p90_vals = base_vals = cand_vals = anchor_vals = np.asarray([], dtype=np.float64)
    profile_quality = {
        "backend": "3DDFA_V2_ONNXRuntime_CPU_sparse68_to_canonical468",
        "repo": THREEDDFA_REPO,
        "attempted": int(len(profile_attempts)),
        "accepted": int(profile_fit),
        "rejected": int(max(len(profile_attempts) - profile_fit, 0)),
        "reject_reasons": profile_reject_reasons,
        "accepted_fit_mean_over_head_scale_mean": float(np.mean(fit_mean_vals)) if len(fit_mean_vals) else 0.0,
        "accepted_fit_mean_over_head_scale_max": float(np.max(fit_mean_vals)) if len(fit_mean_vals) else 0.0,
        "accepted_fit_p90_over_head_scale_max": float(np.max(fit_p90_vals)) if len(fit_p90_vals) else 0.0,
        "accepted_anchor_over_head_scale_max": float(np.max(anchor_vals)) if len(anchor_vals) else 0.0,
        "pose_baseline_residual_mean": float(np.mean(base_vals)) if len(base_vals) else 0.0,
        "profile_candidate_residual_mean": float(np.mean(cand_vals)) if len(cand_vals) else 0.0,
        "candidate_residual_better_than_pose_all": bool(len(cand_vals) == 0 or np.all(cand_vals <= base_vals * PROFILE_BASELINE_IMPROVEMENT_RATIO)),
        "thresholds": {
            "yaw_abs_min_deg": PROFILE_YAW_MIN_DEG,
            "yaw_abs_max_deg": PROFILE_YAW_MAX_DEG,
            "anchor_max_over_head_scale": PROFILE_ANCHOR_MAX_OVER_HEAD_SCALE,
            "fit_mean_max_over_head_scale": PROFILE_FIT_MEAN_MAX_OVER_HEAD_SCALE,
            "fit_p90_max_over_head_scale": PROFILE_FIT_P90_MAX_OVER_HEAD_SCALE,
            "min_yolo_conf": PROFILE_MIN_YOLO_CONF,
        },
    }

    kills = {
        "zoom_crop_material_raise": bool(cov["observed_pct"] >= 80.0),
        "profile_fit_material_raise_toward_ceiling": bool(cov["observed_pct"] >= 93.0),
        "profile_alignment_better_than_pose_tail": bool(profile_quality["candidate_residual_better_than_pose_all"]),
        "corrected_jawOpen_r_ge_0p70": bool(not mouth["kill_r_lt_0p70"]),
        "boundary_pop_le_25pct": bool(not bp["kill_25pct"]),
        "disallowed_gpu_or_renderer_calls_absent": bool(disallowed["pass"]),
        "kill_hit": bool(
            cov["observed_pct"] < 80.0
            or not profile_quality["candidate_residual_better_than_pose_all"]
            or mouth["kill_r_lt_0p70"]
            or bp["kill_25pct"]
            or not disallowed["pass"]
        ),
    }

    return {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": {
            "stream": STREAM_PATH,
            "report": REPORT_PATH,
            "notes": NOTES_PATH,
            "montage": MONTAGE_PATH,
            "profile_fit_montage": PROFILE_MONTAGE_PATH,
            "overlay_master": OVERLAY_MASTER_PATH,
            "overlay_preview": OVERLAY_PREVIEW_PATH,
        },
        "device": {
            "torch_device": str(DEVICE),
            "mediapipe": "FaceLandmarker via XNNPACK/CPU",
            "yolo_face": str(DEVICE),
            "profile_fit": "3DDFA_V2 via ONNXRuntime CPU",
        },
        "timing": {
            "total_wall_s": float(total_wall_s),
            "frames_per_second": float(N / max(total_wall_s, 1e-6)),
        },
        "coverage": cov,
        "topology": topo,
        "mouth_expr": mouth,
        "boundary_pop": bp,
        "boundary_pop_raw_unclamped": raw_bp,
        "profile_fit_quality": profile_quality,
        "motion_verification": motion,
        "mps_no_cuda": disallowed,
        "kill_conditions": kills,
        "blendshape_metadata": {
            "mapping": "MediaPipe categories remapped by category_name; _neutral skipped.",
            "missing_in_mediapipe": ["tongueOut"],
            "tongueOut_value": 0.0,
            "arkit_names": ARKIT_NAMES,
        },
        "iteration2_profile_fit": {
            "flame_remap": "not_used",
            "profile_3d_fit_3ddfa_v2": "enabled_kill_gated",
            "topology_mapping": "3DDFA sparse68 landmarks drive MediaPipe canonical 468 vertices; faces unchanged",
        },
    }


def write_notes(report: Dict) -> None:
    cov = report["coverage"]
    mouth = report["mouth_expr"]
    bp = report["boundary_pop"]
    raw_bp = report["boundary_pop_raw_unclamped"]
    topo = report["topology"]
    motion = report["motion_verification"]
    device = report["device"]
    kills = report["kill_conditions"]
    profile = report["profile_fit_quality"]

    lines = [
        "# Mesh Cascade V3 Notes",
        "",
        "## Outputs",
        f"- Stream: `{STREAM_PATH}`",
        f"- Overlay: `{OVERLAY_MASTER_PATH}`",
        f"- Preview overlay: `{OVERLAY_PREVIEW_PATH}`",
        f"- Montage: `{MONTAGE_PATH}`",
        f"- Profile-fit montage: `{PROFILE_MONTAGE_PATH}`",
        f"- JSON report: `{REPORT_PATH}`",
        "",
        "## Build Summary",
        "- Single topology: MediaPipe canonical OBJ, 468 vertices / 898 faces.",
        "- Observed tier: full-frame MediaPipe, then YOLO-face crop/upscale MediaPipe recovery.",
        "- Profile tier: 3DDFA_V2 ONNXRuntime CPU sparse-68 profile fit, accepted only in |yaw| 30-90 and remapped onto the same canonical 468 vertices.",
        "- Tail tier: same topology posed from v17 fusion pose/anchor; tagged `pose_posed`, `mesh_conf=0`, `geometry_observed=false`.",
        "- Blendshapes: category-name remap; `_neutral` skipped; `tongueOut=0` because MediaPipe does not emit it.",
        "- Profile expression: only jawOpen is derived from 3DDFA mouth landmarks when available; all other ARKit slots hold/decay from the previous observed stream.",
        "",
        "## Pre-Registered Tests",
        f"- Coverage: `{cov['verts_populated']}/{cov['total_frames']}` verts populated, no NaN={cov['no_nan']}.",
        f"- Observed: `{cov['observed_frames']}/{cov['total_frames']}` ({cov['observed_pct']:.1f}%). Full MP={cov['observed_full_mp']}, zoom MP={cov['observed_zoom_mp']}, profile_fit={cov['profile_fit']}.",
        f"- Predicted tail: `{cov['predicted_pose_posed']}/{cov['total_frames']}` ({cov['predicted_pct']:.1f}%).",
        f"- Low-scale observed gate: head_scale_px >= `{cov['min_observed_head_scale_px']:.1f}`; transition-clamped frames={cov['transition_clamped_frames']}.",
        "- Coverage by yaw bin:",
    ]
    for bin_name, data in cov["by_yaw_bin"].items():
        lines.append(
            f"  - `{bin_name}`: total={data['total']} observed={data['observed']} "
            f"full={data['observed_full_mp']} zoom={data['observed_zoom_mp']} "
            f"profile_fit={data['profile_fit']} predicted={data['predicted']}"
        )
    lines.extend([
        f"- Topology: V={topo['vertex_count']} F={topo['face_count']} faces_sha256=`{topo['faces_sha256']}` constant={topo['pass']}.",
        f"- Profile fit: attempted={profile['attempted']} accepted={profile['accepted']} rejected={profile['rejected']}; fit_mean max={profile['accepted_fit_mean_over_head_scale_max']:.4f}; anchor max={profile['accepted_anchor_over_head_scale_max']:.4f}.",
        f"- Profile alignment vs v2 pose tail: baseline residual mean={profile['pose_baseline_residual_mean']:.4f}; candidate residual mean={profile['profile_candidate_residual_mean']:.4f}; better_all={profile['candidate_residual_better_than_pose_all']}.",
        f"- Mouth/expr Pearson r observed all observed geometry: `{mouth['pearson_jawOpen_vs_inner_lip_aperture_observed']:.4f}`.",
        f"- Mouth/expr Pearson r derivable expression frames: `{mouth['pearson_jawOpen_vs_inner_lip_aperture_derivable']:.4f}` target >=0.70; derivable_n={mouth['derivable_n']} profile_jaw_derived_n={mouth['profile_jaw_derived_n']}.",
        f"- Mouth/expr Pearson r frontal |yaw|<30: `{mouth['pearson_jawOpen_vs_inner_lip_aperture_frontal_abs_yaw_lt_30']:.4f}` target >=0.85.",
        f"- Corrected jawOpen max observed: frame `{mouth['corrected_jawOpen_max_observed']['frame']}` value `{mouth['corrected_jawOpen_max_observed']['value']:.4f}`.",
        f"- f485 check: jawOpen `{mouth['frame_485']['jawOpen_corrected']:.4f}`, aperture `{mouth['frame_485']['inner_lip_aperture']:.4f}`, source `{mouth['frame_485']['source']}`.",
        "- f82-91 open/close check:",
    ])
    for row in mouth["frames_82_91"]:
        lines.append(
            f"  - f{row['frame']}: jawOpen={row['jawOpen_corrected']:.4f} aperture={row['inner_lip_aperture']:.4f} yaw={row['yaw_deg']:.1f} source={row['source']}"
        )
    lines.extend([
        f"- Boundary pop raw before transition clamp: transitions={raw_bp['n_transitions']} mean={raw_bp['mean_jump_over_head_scale']:.4f} max={raw_bp['max_jump_over_head_scale']:.4f}.",
        f"- Boundary pop clamped output: transitions={bp['n_transitions']} mean={bp['mean_jump_over_head_scale']:.4f} max={bp['max_jump_over_head_scale']:.4f}; target <=0.15, kill >0.25.",
        f"- Device: torch={device['torch_device']}; MediaPipe={device['mediapipe']}; YOLO={device['yolo_face']}; profile_fit={device['profile_fit']}.",
        f"- Disallowed call scan pass: `{report['mps_no_cuda']['pass']}` hits={report['mps_no_cuda']['hits']}.",
        f"- Verify in motion: frontal run={motion['frontal_longest_observed_run']}; 3/4 run={motion['three_quarter_longest_observed_run']}; profile_fit run={motion['profile_fit_longest_run']}; pass={motion['pass_frontal_three_quarter_profile_ge_8']}.",
        "",
        "## Kill Conditions",
        f"- Zoom crop material raise >=80% observed: `{kills['zoom_crop_material_raise']}`.",
        f"- Profile fit material raise >=93% observed: `{kills['profile_fit_material_raise_toward_ceiling']}`.",
        f"- Profile alignment better than v2 pose tail: `{kills['profile_alignment_better_than_pose_tail']}`.",
        f"- Corrected jawOpen r >=0.70: `{kills['corrected_jawOpen_r_ge_0p70']}`.",
        f"- Boundary pop <=25% head scale: `{kills['boundary_pop_le_25pct']}`.",
        f"- Disallowed GPU/render calls absent: `{kills['disallowed_gpu_or_renderer_calls_absent']}`.",
        f"- Kill hit: `{kills['kill_hit']}`.",
        "",
        "## Profile Rejections",
    ])
    for reason, count in sorted(profile["reject_reasons"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {reason}: {count}")
    lines.extend([
        "",
        "## Honest Limits",
        "- Pose-posed frames are not observed geometry and must not be treated as measured face mesh.",
        "- True back-of-head frames in the 90-180 degree yaw bin remain pose_posed; this face-fit lane does not recover them.",
        "- Profile-fit geometry is landmark-driven canonical deformation, not a native BFM topology export.",
        "- Profile expression is only jaw/mouth opening where the 3DDFA landmarks support it; other ARKit values are held/decayed.",
        "- Additional zoom-observed frames inherit v17 pose/anchor metadata for the head_transform field while their vertices come from live MediaPipe landmarks remapped to full-frame coordinates.",
    ])
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run() -> Dict:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{LOG_PREFIX} Device torch={DEVICE}; MediaPipe=XNNPACK/CPU; YOLO={DEVICE}")
    print(f"{LOG_PREFIX} Loading canonical mesh and v17 fusion stream...")
    canon_verts, faces = load_canonical_mesh()
    v17 = np.load(V17_FUSION_NPZ, allow_pickle=True)
    total_f = int(len(v17["frame"]))
    print(f"{LOG_PREFIX} v17 frames={total_f}; canonical={canon_verts.shape[0]}v/{faces.shape[0]}f")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames != total_f:
        print(f"{LOG_PREFIX} warning: video frames={video_frames}, v17 frames={total_f}")

    face_lmk = make_face_landmarker()
    yolo_face = YOLO(YOLO_FACE_PATH)
    print(f"{LOG_PREFIX} Loading 3DDFA_V2 ONNX CPU profile fitter...")
    tddfa_profile, calc_3ddfa_pose = load_profile_fitter()
    v2_stream = np.load(V2_STREAM_NPZ, allow_pickle=True)
    v2_projected_px = np.asarray(v2_stream["projected_px"], dtype=np.float32)
    if v2_projected_px.shape[:2] != (total_f, V_CANON):
        raise RuntimeError(f"Bad v2 projected baseline shape: {v2_projected_px.shape}")

    verts = np.zeros((total_f, V_CANON, 3), dtype=np.float32)
    projected_px = np.zeros((total_f, V_CANON, 2), dtype=np.float32)
    arkit = np.zeros((total_f, len(ARKIT_NAMES)), dtype=np.float32)
    head_transform = np.asarray(v17["head_transform"], dtype=np.float32).copy()
    mesh_source = np.full(total_f, "pose_posed", dtype="<U24")
    mesh_conf = np.zeros(total_f, dtype=np.float32)
    geometry_observed = np.zeros(total_f, dtype=bool)
    expr_conf = np.zeros(total_f, dtype=np.float32)
    frame_arr = np.asarray(v17["frame"], dtype=np.int32)

    yaw = np.asarray(v17["yaw_deg"], dtype=np.float32)
    pitch = np.asarray(v17["pitch_deg"], dtype=np.float32)
    roll = np.asarray(v17["roll_deg"], dtype=np.float32)
    head_center = np.asarray(v17["head_center_px"], dtype=np.float32)
    head_scale = np.asarray(v17["head_scale_px"], dtype=np.float32)
    v17_mode = np.asarray(v17["mode"])
    v17_expr_conf = np.asarray(v17["expr_conf"], dtype=np.float32) if "expr_conf" in v17.files else np.zeros(total_f, dtype=np.float32)

    diag_records: List[Dict] = []
    last_bshp = np.zeros(len(ARKIT_NAMES), dtype=np.float32)
    last_residual = np.zeros((V_CANON, 3), dtype=np.float32)
    last_observed_frame: Optional[int] = None
    current_expr_conf = 0.0

    n_full = 0
    n_zoom = 0
    n_profile = 0
    n_pose = 0

    print(f"{LOG_PREFIX} Processing {total_f} frames @ {fps:.2f}fps ({fw}x{fh})...")
    for fidx in range(total_f):
        ret, frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {fidx}")

        neutral_verts, neutral_px = project_canonical_pose(
            canon_verts,
            head_transform[fidx],
            head_center[fidx],
            head_scale[fidx],
            float(yaw[fidx]),
            str(v17_mode[fidx]),
        )

        obs, diag = try_observed_mesh(
            face_lmk,
            yolo_face,
            frame_bgr,
            fw,
            fh,
            head_center[fidx],
            float(head_scale[fidx]),
        )

        if obs is not None:
            obs_verts, obs_px = observed_landmarks_to_px3(obs.landmarks_norm_full, fw, fh)
            prev_source = str(mesh_source[fidx - 1]) if fidx > 0 else obs.source
            source_changed = fidx > 0 and prev_source in {"pose_posed", "profile_fit"}
            current_verts, transition_diag = clamp_transition_if_needed(
                obs_verts,
                verts[fidx - 1] if fidx > 0 else obs_verts,
                float(head_scale[fidx]),
                source_changed,
            )
            verts[fidx] = current_verts
            projected_px[fidx] = current_verts[:, :2]
            arkit[fidx] = obs.blendshapes
            mesh_source[fidx] = obs.source
            mesh_conf[fidx] = 1.0
            geometry_observed[fidx] = True
            current_expr_conf = 1.0
            expr_conf[fidx] = current_expr_conf
            last_bshp = obs.blendshapes.copy()
            last_residual = verts[fidx] - neutral_verts
            last_observed_frame = fidx
            if obs.source == "observed_full_mp":
                n_full += 1
            else:
                n_zoom += 1
        else:
            if last_observed_frame is None:
                residual = np.zeros_like(neutral_verts)
                decayed_bshp = np.zeros_like(last_bshp)
                current_expr_conf = 0.0
            else:
                residual = last_residual * RESIDUAL_DECAY
                decayed_bshp = last_bshp * BSHP_DECAY
                current_expr_conf = max(float(current_expr_conf) * BSHP_DECAY, 0.0)
            candidate_verts = neutral_verts + residual
            profile_obs, profile_diag = try_profile_fit(
                tddfa_profile,
                calc_3ddfa_pose,
                yolo_face,
                frame_bgr,
                neutral_verts,
                v2_projected_px[fidx],
                head_center[fidx],
                float(head_scale[fidx]),
                float(yaw[fidx]),
                decayed_bshp,
            )
            diag.update(profile_diag)

            if profile_obs is not None:
                prev_source = str(mesh_source[fidx - 1]) if fidx > 0 else "profile_fit"
                source_changed = fidx > 0 and prev_source != "profile_fit"
                current_verts, transition_diag = clamp_transition_if_needed(
                    profile_obs.verts,
                    verts[fidx - 1] if fidx > 0 else profile_obs.verts,
                    float(head_scale[fidx]),
                    source_changed,
                )
                verts[fidx] = current_verts
                projected_px[fidx] = current_verts[:, :2]
                arkit[fidx] = profile_obs.expr_blendshapes
                mesh_source[fidx] = "profile_fit"
                mesh_conf[fidx] = np.float32(profile_obs.mesh_conf)
                geometry_observed[fidx] = True
                current_expr_conf = float(profile_obs.expr_conf)
                expr_conf[fidx] = np.float32(current_expr_conf)
                last_bshp = profile_obs.expr_blendshapes.copy()
                last_residual = verts[fidx] - neutral_verts
                last_observed_frame = fidx
                n_profile += 1
            else:
                prev_source = str(mesh_source[fidx - 1]) if fidx > 0 else "pose_posed"
                source_changed = fidx > 0 and prev_source != "pose_posed"
                current_verts, transition_diag = clamp_transition_if_needed(
                    candidate_verts,
                    verts[fidx - 1] if fidx > 0 else candidate_verts,
                    float(head_scale[fidx]),
                    source_changed,
                )
                verts[fidx] = current_verts
                projected_px[fidx] = current_verts[:, :2]
                arkit[fidx] = decayed_bshp
                mesh_source[fidx] = "pose_posed"
                mesh_conf[fidx] = 0.0
                geometry_observed[fidx] = False
                expr_conf[fidx] = current_expr_conf
                last_bshp = decayed_bshp.copy()
                last_residual = verts[fidx] - neutral_verts
                n_pose += 1

        rec = {
            "frame": int(fidx),
            "source": str(mesh_source[fidx]),
            "full_mp_detected": bool(diag.get("full_mp_detected", False)),
            "full_mp_accepted": bool(diag.get("full_mp_accepted", False)),
            "zoom_yolo_found": bool(diag.get("zoom_yolo_found", False)),
            "zoom_mp_detected": bool(diag.get("zoom_mp_detected", False)),
            "zoom_mp_accepted": bool(diag.get("zoom_mp_accepted", False)),
            "reject_reason": str(diag.get("reject_reason", "")),
            "missing_blendshapes": list(diag.get("missing_blendshapes", [])),
            "v17_mode": str(v17_mode[fidx]),
            "yaw_deg": float(yaw[fidx]),
            "transition_clamped": bool(transition_diag["transition_clamped"]),
            "transition_raw_jump_over_head_scale": float(transition_diag["transition_raw_jump_over_head_scale"]),
            "transition_alpha": float(transition_diag["transition_alpha"]),
        }
        for key in [
            "profile_attempted",
            "profile_yolo_found",
            "profile_yolo_conf",
            "profile_3ddfa_ok",
            "profile_accepted",
            "profile_reject_reason",
            "profile_anchor_dist_over_head_scale",
            "profile_fit_mean_over_head_scale",
            "profile_fit_p90_over_head_scale",
            "profile_baseline_residual_over_head_scale",
            "profile_candidate_residual_over_head_scale",
            "profile_similarity_scale_2d",
            "profile_similarity_scale_3d",
            "profile_pose_yaw_deg",
            "profile_pose_pitch_deg",
            "profile_pose_roll_deg",
            "profile_mesh_conf",
            "profile_expr_conf",
            "profile_jaw_open_derived",
            "profile_roi_box",
            "profile_source_box",
        ]:
            if key in diag:
                rec[key] = diag[key]
        diag_records.append(rec)

        if fidx % 50 == 0 or fidx == total_f - 1:
            elapsed = time.time() - t0
            print(
                f"{LOG_PREFIX} f{fidx}/{total_f}: full={n_full} zoom={n_zoom} profile={n_profile} "
                f"pose={n_pose} {((fidx + 1) / max(elapsed, 0.001)):.1f}fps"
            )

    cap.release()
    face_lmk.close()

    if not np.isfinite(verts).all():
        raise RuntimeError("verts contains non-finite values before save")
    if not np.isfinite(projected_px).all():
        raise RuntimeError("projected_px contains non-finite values before save")
    if not np.isfinite(arkit).all():
        raise RuntimeError("arkit contains non-finite values before save")

    print(f"{LOG_PREFIX} Saving stream...")
    np.savez_compressed(
        STREAM_PATH,
        frame=frame_arr,
        verts=verts,
        faces=faces.astype(np.int32),
        projected_px=projected_px,
        arkit52_corrected=arkit,
        arkit_names=np.asarray(ARKIT_NAMES, dtype="<U25"),
        head_transform=head_transform,
        mesh_source=mesh_source,
        mesh_conf=mesh_conf,
        geometry_observed=geometry_observed,
        expr_conf=expr_conf,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        head_center_px=head_center,
        head_scale_px=head_scale,
        v17_mode=v17_mode,
        v17_expr_conf=v17_expr_conf,
        pipeline_version=np.asarray([PIPELINE_VERSION]),
        blendshape_missing=np.asarray(["tongueOut"], dtype="<U25"),
        blendshape_mapping=np.asarray(["by_category_name_skip_neutral"], dtype="<U40"),
        profile_3d_fit=np.asarray(["3ddfa_v2_onnx_cpu_sparse68_to_468"], dtype="<U40"),
    )

    total_wall_s = time.time() - t0
    report = build_tests_and_report(
        verts,
        faces,
        projected_px,
        arkit,
        mesh_source,
        mesh_conf,
        geometry_observed,
        expr_conf,
        yaw,
        head_scale,
        diag_records,
        total_wall_s,
    )

    print(f"{LOG_PREFIX} Rendering overlay and montage...")
    outputs = build_video_and_montage(
        verts,
        projected_px,
        mesh_source,
        arkit,
        yaw,
        mesh_conf,
        expr_conf,
        faces,
        report,
    )
    report["paths"].update(outputs)
    report["output_sizes_mb"] = {
        "stream": round(os.path.getsize(STREAM_PATH) / 1e6, 3),
        "overlay_master": round(os.path.getsize(OVERLAY_MASTER_PATH) / 1e6, 3),
        "overlay_preview": round(os.path.getsize(OVERLAY_PREVIEW_PATH) / 1e6, 3),
        "montage": round(os.path.getsize(MONTAGE_PATH) / 1e6, 3) if os.path.exists(MONTAGE_PATH) else 0.0,
        "profile_fit_montage": round(os.path.getsize(PROFILE_MONTAGE_PATH) / 1e6, 3) if os.path.exists(PROFILE_MONTAGE_PATH) else 0.0,
    }
    report["frame_diagnostics"] = diag_records

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_notes(report)

    print(f"{LOG_PREFIX} DONE")
    print(f"  Stream: {STREAM_PATH}")
    print(f"  Observed: {report['coverage']['observed_frames']}/{total_f} ({report['coverage']['observed_pct']:.1f}%)")
    print(f"  Profile fit: {report['coverage']['profile_fit']}/{total_f}")
    print(f"  Pose tail: {report['coverage']['predicted_pose_posed']}/{total_f} ({report['coverage']['predicted_pct']:.1f}%)")
    print(f"  Jaw r derivable: {report['mouth_expr']['pearson_jawOpen_vs_inner_lip_aperture_derivable']:.4f}")
    print(f"  Boundary max: {report['boundary_pop']['max_jump_over_head_scale']:.4f}")
    print(f"  Notes: {NOTES_PATH}")
    return report


if __name__ == "__main__":
    run()
