#!/usr/bin/env python3
"""
mesh_avatar_driver_v1.py

End-to-end proof that mesh_cascade_v4 geometry drives the emoji_head.glb rig.

Primary driver:
  mesh_cascade_v4_stream.npz projected_px / verts only.

Audit-only:
  arkit52_corrected and head_transform are loaded only for comparison metrics.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

from avatar_overlay_pipeline_v16 import (
    EmojiGLB,
    AvatarRenderer,
    build_T_from_screen_pos,
    composite_rgba_bgr,
    FOCAL_LEN,
)


OUT_DIR = "."
VIDEO_PATH = "input_clip.mov"
STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v4_stream.npz"
CANONICAL_OBJ = "assets/canonical_face_model.obj"
GLB_PATH = "assets/emoji_head.glb"

RAW_PATH = f"{OUT_DIR}/mesh_avatar_driver_v1_raw.mp4"
MASTER_PATH = f"{OUT_DIR}/mesh_avatar_driver_v1_master.mp4"
PREVIEW_PATH = f"{OUT_DIR}/mesh_avatar_driver_v1_preview.mp4"
MONTAGE_PATH = f"{OUT_DIR}/mesh_avatar_driver_v1_montage.png"
REPORT_PATH = f"{OUT_DIR}/mesh_avatar_driver_v1_report.json"
NOTES_PATH = f"{OUT_DIR}/notes_mesh_avatar_driver_v1.md"
CONTROL_STREAM_PATH = f"{OUT_DIR}/mesh_avatar_driver_v1_controls.npz"

PIPELINE_VERSION = "mesh_avatar_driver_v1"
EXPECTED_STREAM_VERSION = "mesh_cascade_v4"
LOG_PREFIX = "[mesh-avatar-driver-v1]"

Z_REF = -88.8
AVATAR_SCALE_K = 0.065
AVATAR_SCALE_MIN = 3.0
AVATAR_SCALE_MAX = 25.0
PANEL_W = 360
PANEL_H = 640

V_CANON = 468
MP_EAR_L = 234
MP_EAR_R = 454

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

ESSENTIAL_TARGETS = [
    "jawOpen",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "browInnerUp",
    "jawLeft",
    "jawRight",
    "mouthPucker",
    "mouthFunnel",
]

DRIVER_TARGETS = [
    "jawOpen",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFunnel",
    "mouthPucker",
    "mouthFrownLeft",
    "mouthFrownRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "browDownLeft",
    "browDownRight",
    "cheekPuff",
    "eyeWideLeft",
    "eyeWideRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "mouthLeft",
    "mouthRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "jawLeft",
    "jawRight",
]

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
LEFT_BROW = [
    (46, 53), (53, 52), (52, 65), (65, 55), (55, 70), (70, 63), (63, 105),
    (105, 66), (66, 107), (107, 46),
]
RIGHT_BROW = [
    (276, 283), (283, 282), (282, 295), (295, 285), (285, 300), (300, 293),
    (293, 334), (334, 296), (296, 336), (336, 276),
]
FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]

MOUTH_KEY_LMK = [
    0, 13, 14, 17, 61, 78, 87, 88, 91, 146, 181,
    267, 269, 270, 291, 308, 311, 312, 314, 317, 318,
    321, 324, 375, 402, 405, 409, 415,
]

PNP_IDX = np.asarray(sorted(set([
    1, 4, 6, 10, 152, 234, 454, 33, 133, 362, 263, 159, 386,
    61, 291, 13, 14, 17, 0, 78, 308, 70, 105, 336, 334,
    93, 323, 127, 356, 168, 197, 2, 98, 327,
] + list(range(0, V_CANON, 9)))), dtype=np.int32)

SOURCE_COLORS_BGR = {
    "observed": (30, 210, 70),
    "profile": (0, 180, 255),
    "interpolated": (255, 130, 20),
}
SOURCE_TINT_RGB = {
    "observed": None,
    "profile": np.asarray([1.0, 0.62, 0.08], dtype=np.float32),
    "interpolated": np.asarray([0.10, 0.45, 1.0], dtype=np.float32),
}


@dataclass
class MeshStream:
    frame: np.ndarray
    verts: np.ndarray
    faces: np.ndarray
    projected_px: np.ndarray
    arkit52: np.ndarray
    arkit_names: List[str]
    head_transform: np.ndarray
    mesh_source: np.ndarray
    mesh_conf: np.ndarray
    geometry_observed: np.ndarray
    expr_conf: np.ndarray
    stream_yaw: np.ndarray
    stream_pitch: np.ndarray
    stream_roll: np.ndarray


@dataclass
class Calibration:
    neutral_mask: np.ndarray
    feature_base: Dict[str, float]
    feature_lo: Dict[str, float]
    feature_hi: Dict[str, float]
    neutral_frame_count: int
    identity_residual: Dict[str, float]


@dataclass
class PoseTracks:
    center_raw: np.ndarray
    scale_raw: np.ndarray
    center_smooth: np.ndarray
    scale_smooth: np.ndarray
    rotations: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    pose_ok: np.ndarray


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def source_class(source: str) -> str:
    s = str(source)
    if s == "interpolated":
        return "interpolated"
    if s == "profile_fit":
        return "profile"
    return "observed"


def load_stream() -> MeshStream:
    z = np.load(STREAM_PATH, allow_pickle=True)
    pipeline_version = str(z["pipeline_version"][0])
    assert pipeline_version == EXPECTED_STREAM_VERSION, (
        f"Expected {EXPECTED_STREAM_VERSION}, got {pipeline_version}"
    )
    return MeshStream(
        frame=np.asarray(z["frame"], dtype=np.int32),
        verts=np.asarray(z["verts"], dtype=np.float32),
        faces=np.asarray(z["faces"], dtype=np.int32),
        projected_px=np.asarray(z["projected_px"], dtype=np.float32),
        arkit52=np.asarray(z["arkit52_corrected"], dtype=np.float32),
        arkit_names=[str(x) for x in z["arkit_names"]],
        head_transform=np.asarray(z["head_transform"], dtype=np.float32),
        mesh_source=np.asarray(z["mesh_source"]).astype(str),
        mesh_conf=np.asarray(z["mesh_conf"], dtype=np.float32),
        geometry_observed=np.asarray(z["geometry_observed"], dtype=bool),
        expr_conf=np.asarray(z["expr_conf"], dtype=np.float32),
        stream_yaw=np.asarray(z["yaw_deg"], dtype=np.float32),
        stream_pitch=np.asarray(z["pitch_deg"], dtype=np.float32),
        stream_roll=np.asarray(z["roll_deg"], dtype=np.float32),
    )


def video_meta() -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, fw, fh, n


def faces_to_edges(faces: np.ndarray) -> List[Tuple[int, int]]:
    edges = set()
    for tri in np.asarray(faces, dtype=np.int32):
        a, b, c = [int(x) for x in tri]
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    return sorted(edges)


def draw_text(img: np.ndarray, text: str, xy: Tuple[int, int],
              color: Tuple[int, int, int], scale: float = 0.48,
              thickness: int = 1) -> None:
    x, y = xy
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_wireframe(frame_bgr: np.ndarray, points_px: np.ndarray, faces: np.ndarray,
                   source: str, yaw: float, jaw: float) -> np.ndarray:
    out = frame_bgr.copy()
    cls = source_class(source)
    col = SOURCE_COLORS_BGR[cls]
    edges = faces_to_edges(faces)
    p = np.asarray(points_px, dtype=np.float32)

    for a, b in edges:
        pa = (int(round(p[a, 0])), int(round(p[a, 1])))
        pb = (int(round(p[b, 0])), int(round(p[b, 1])))
        cv2.line(out, pa, pb, (80, 80, 80), 1, cv2.LINE_AA)

    for contour, contour_col, lw in [
        (FACE_OVAL, col, 2),
        (LEFT_BROW + RIGHT_BROW, (220, 60, 220), 2),
        (LEFT_EYE + RIGHT_EYE, (255, 160, 40), 2),
        (LIPS_INNER, (0, 230, 255), 2),
        (LIPS_OUTER, (0, 110, 255), 3),
    ]:
        for a, b in contour:
            pa = (int(round(p[a, 0])), int(round(p[a, 1])))
            pb = (int(round(p[b, 0])), int(round(p[b, 1])))
            cv2.line(out, pa, pb, contour_col, lw, cv2.LINE_AA)

    for i in MOUTH_KEY_LMK:
        cv2.circle(out, (int(round(p[i, 0])), int(round(p[i, 1]))),
                   2, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), col, 6)
    draw_text(out, f"v4 mesh wire | {source} | yaw={yaw:+.0f} jaw={jaw:.2f}",
              (14, 34), col, 0.62, 1)
    return out


def panelize(real: np.ndarray, wire: np.ndarray, avatar: np.ndarray) -> np.ndarray:
    cells = []
    for img in [real, wire, avatar]:
        cells.append(cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA))
    return np.concatenate(cells, axis=1)


def mesh_center_scale(points_px: np.ndarray) -> Tuple[np.ndarray, float]:
    p = np.asarray(points_px, dtype=np.float64)
    eye_mid = 0.5 * (p[159] + p[386])
    ear_mid = 0.5 * (p[MP_EAR_L] + p[MP_EAR_R])
    center = 0.75 * eye_mid + 0.25 * ear_mid
    face_h = float(np.linalg.norm(p[10] - p[152]))
    ear_span = float(np.linalg.norm(p[MP_EAR_L] - p[MP_EAR_R]))
    scale = max(0.78 * face_h, 0.55 * ear_span, 20.0)
    return center.astype(np.float64), float(scale)


def feature_values(points_px: np.ndarray) -> Dict[str, float]:
    p = np.asarray(points_px, dtype=np.float64)
    face_h = max(float(np.linalg.norm(p[10] - p[152])), 1e-6)

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(p[a] - p[b]))

    mouth_center = 0.5 * (p[13] + p[14])
    mouth_width = dist(61, 291) / face_h
    mouth_ap = dist(13, 14) / face_h
    mouth_outer_ap = dist(0, 17) / face_h
    smile_l = (mouth_center[1] - p[61, 1]) / face_h
    smile_r = (mouth_center[1] - p[291, 1]) / face_h
    frown_l = (p[61, 1] - mouth_center[1]) / face_h
    frown_r = (p[291, 1] - mouth_center[1]) / face_h
    jaw_x = (0.5 * (p[14, 0] + p[17, 0]) - p[4, 0]) / face_h
    mouth_x = (mouth_center[0] - p[4, 0]) / face_h

    eye_l_w = max(dist(33, 133), 1e-6)
    eye_r_w = max(dist(362, 263), 1e-6)
    eye_l = np.mean([dist(159, 145), dist(158, 153), dist(157, 154)]) / eye_l_w
    eye_r = np.mean([dist(386, 374), dist(385, 380), dist(387, 373)]) / eye_r_w

    brow_l_y = float(np.mean(p[[70, 63, 105, 66, 107], 1]))
    brow_r_y = float(np.mean(p[[336, 296, 334, 293, 300], 1]))
    eye_l_y = float(np.mean(p[[33, 133, 159, 145], 1]))
    eye_r_y = float(np.mean(p[[362, 263, 386, 374], 1]))
    brow_l = (eye_l_y - brow_l_y) / face_h
    brow_r = (eye_r_y - brow_r_y) / face_h

    return {
        "face_h": face_h,
        "mouth_aperture": mouth_ap,
        "mouth_outer_aperture": mouth_outer_ap,
        "mouth_width": mouth_width,
        "smile_l": smile_l,
        "smile_r": smile_r,
        "frown_l": frown_l,
        "frown_r": frown_r,
        "jaw_x": jaw_x,
        "mouth_x": mouth_x,
        "eye_open_l": eye_l,
        "eye_open_r": eye_r,
        "brow_clear_l": brow_l,
        "brow_clear_r": brow_r,
    }


def percentile_dict(features: List[Dict[str, float]], q: float) -> Dict[str, float]:
    keys = list(features[0].keys())
    return {
        k: float(np.nanpercentile([f[k] for f in features], q))
        for k in keys
    }


def kabsch_residual(canonical: np.ndarray, neutral: np.ndarray) -> Dict[str, float]:
    canon = np.asarray(canonical, dtype=np.float64).copy()
    neu = np.asarray(neutral, dtype=np.float64).copy()
    neu[:, 1] *= -1.0
    canon -= 0.5 * (canon[MP_EAR_L] + canon[MP_EAR_R])
    neu -= 0.5 * (neu[MP_EAR_L] + neu[MP_EAR_R])
    canon /= max(float(np.linalg.norm(canon[MP_EAR_L] - canon[MP_EAR_R])), 1e-9)
    neu /= max(float(np.linalg.norm(neu[MP_EAR_L] - neu[MP_EAR_R])), 1e-9)
    A = canon - np.mean(canon, axis=0)
    B = neu - np.mean(neu, axis=0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    Ar = A @ R.T
    s = float((Ar * B).sum() / max(float((Ar * Ar).sum()), 1e-9))
    residual = np.linalg.norm(s * Ar - B, axis=1)
    return {
        "method": "neutral_verts_vs_canonical_similarity_fit_normalized_by_ear_span",
        "mean": float(np.mean(residual)),
        "median": float(np.median(residual)),
        "p95": float(np.percentile(residual, 95.0)),
        "scale": s,
    }


def calibrate(stream: MeshStream, pose_yaw: np.ndarray,
              canonical: np.ndarray) -> Calibration:
    observed = np.char.startswith(stream.mesh_source.astype(str), "observed")
    neutral_mask = (
        observed
        & (stream.mesh_conf >= 0.95)
        & (np.abs(pose_yaw) < 8.0)
    )
    if int(neutral_mask.sum()) < 20:
        neutral_mask = (
            observed
            & (stream.mesh_conf >= 0.90)
            & (np.abs(pose_yaw) < 15.0)
        )
    if int(neutral_mask.sum()) < 10:
        raise RuntimeError("Not enough frontal high-confidence observed frames for neutral calibration")

    neutral_features = [feature_values(stream.projected_px[i]) for i in np.where(neutral_mask)[0]]
    all_observed_features = [feature_values(stream.projected_px[i]) for i in np.where(observed)[0]]
    base = percentile_dict(neutral_features, 50.0)
    lo = percentile_dict(all_observed_features, 2.0)
    hi = percentile_dict(all_observed_features, 98.0)
    neutral_verts = np.median(stream.verts[neutral_mask], axis=0)
    identity_res = kabsch_residual(canonical, neutral_verts)
    return Calibration(
        neutral_mask=neutral_mask,
        feature_base=base,
        feature_lo=lo,
        feature_hi=hi,
        neutral_frame_count=int(neutral_mask.sum()),
        identity_residual=identity_res,
    )


def norm_hi(value: float, base: float, hi: float, min_span: float) -> float:
    return clip01((value - base) / max(hi - base, min_span))


def norm_lo(value: float, base: float, lo: float, min_span: float) -> float:
    return clip01((base - value) / max(base - lo, min_span))


def raw_controls_from_features(feat: Dict[str, float], cal: Calibration) -> Dict[str, float]:
    b = cal.feature_base
    lo = cal.feature_lo
    hi = cal.feature_hi

    jaw = norm_hi(
        max(feat["mouth_aperture"], 0.55 * feat["mouth_outer_aperture"]),
        max(b["mouth_aperture"], 0.55 * b["mouth_outer_aperture"]),
        max(hi["mouth_aperture"], 0.55 * hi["mouth_outer_aperture"]),
        0.030,
    )
    smile_l = norm_hi(feat["smile_l"], b["smile_l"], hi["smile_l"], 0.020)
    smile_r = norm_hi(feat["smile_r"], b["smile_r"], hi["smile_r"], 0.020)
    frown_l = norm_hi(feat["frown_l"], b["frown_l"], hi["frown_l"], 0.020)
    frown_r = norm_hi(feat["frown_r"], b["frown_r"], hi["frown_r"], 0.020)

    blink_l = norm_lo(feat["eye_open_l"], b["eye_open_l"], lo["eye_open_l"], 0.035)
    blink_r = norm_lo(feat["eye_open_r"], b["eye_open_r"], lo["eye_open_r"], 0.035)
    wide_l = norm_hi(feat["eye_open_l"], b["eye_open_l"], hi["eye_open_l"], 0.025)
    wide_r = norm_hi(feat["eye_open_r"], b["eye_open_r"], hi["eye_open_r"], 0.025)

    brow_l_up = norm_hi(feat["brow_clear_l"], b["brow_clear_l"], hi["brow_clear_l"], 0.020)
    brow_r_up = norm_hi(feat["brow_clear_r"], b["brow_clear_r"], hi["brow_clear_r"], 0.020)
    brow_l_down = norm_lo(feat["brow_clear_l"], b["brow_clear_l"], lo["brow_clear_l"], 0.018)
    brow_r_down = norm_lo(feat["brow_clear_r"], b["brow_clear_r"], lo["brow_clear_r"], 0.018)

    pucker = norm_lo(feat["mouth_width"], b["mouth_width"], lo["mouth_width"], 0.040)
    stretch = norm_hi(feat["mouth_width"], b["mouth_width"], hi["mouth_width"], 0.055)
    funnel = clip01(0.55 * pucker + 0.40 * jaw)

    jaw_delta = feat["jaw_x"] - b["jaw_x"]
    mouth_delta = feat["mouth_x"] - b["mouth_x"]
    lateral = 0.65 * jaw_delta + 0.35 * mouth_delta
    jaw_right = clip01(lateral / 0.055)
    jaw_left = clip01(-lateral / 0.055)

    values = {name: 0.0 for name in DRIVER_TARGETS}
    values.update({
        "jawOpen": 0.92 * jaw,
        "mouthSmileLeft": 0.80 * smile_l,
        "mouthSmileRight": 0.80 * smile_r,
        "mouthFrownLeft": 0.55 * frown_l,
        "mouthFrownRight": 0.55 * frown_r,
        "mouthPucker": 0.75 * pucker,
        "mouthFunnel": 0.70 * funnel,
        "mouthStretchLeft": 0.55 * stretch,
        "mouthStretchRight": 0.55 * stretch,
        "eyeBlinkLeft": 0.95 * blink_l,
        "eyeBlinkRight": 0.95 * blink_r,
        "eyeWideLeft": 0.70 * wide_l,
        "eyeWideRight": 0.70 * wide_r,
        "eyeSquintLeft": 0.25 * blink_l,
        "eyeSquintRight": 0.25 * blink_r,
        "browInnerUp": 0.50 * max(brow_l_up, brow_r_up) + 0.25 * (brow_l_up + brow_r_up),
        "browOuterUpLeft": 0.70 * brow_l_up,
        "browOuterUpRight": 0.70 * brow_r_up,
        "browDownLeft": 0.70 * brow_l_down,
        "browDownRight": 0.70 * brow_r_down,
        "jawLeft": 0.75 * jaw_left,
        "jawRight": 0.75 * jaw_right,
        "mouthLeft": 0.55 * jaw_left,
        "mouthRight": 0.55 * jaw_right,
        "cheekPuff": 0.10 * pucker,
    })
    return {k: clip01(v) for k, v in values.items()}


def derive_raw_controls(stream: MeshStream, cal: Calibration) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    controls = np.zeros((len(stream.frame), len(DRIVER_TARGETS)), dtype=np.float32)
    feature_rows: List[Dict[str, float]] = []
    for i in range(len(stream.frame)):
        feat = feature_values(stream.projected_px[i])
        vals = raw_controls_from_features(feat, cal)
        feature_rows.append(feat)
        for j, name in enumerate(DRIVER_TARGETS):
            controls[i, j] = np.float32(vals.get(name, 0.0))
    return controls, feature_rows


def retarget_controls_by_source(raw: np.ndarray, sources: np.ndarray) -> np.ndarray:
    out = np.zeros_like(raw, dtype=np.float32)
    prev = np.zeros(raw.shape[1], dtype=np.float32)
    idx = {name: i for i, name in enumerate(DRIVER_TARGETS)}
    for i in range(raw.shape[0]):
        cls = source_class(str(sources[i]))
        if cls == "observed":
            target = raw[i].copy()
            alpha = 0.80
        elif cls == "profile":
            target = prev * 0.90
            for name in ["jawOpen", "mouthPucker", "mouthFunnel", "jawLeft", "jawRight", "mouthLeft", "mouthRight"]:
                j = idx[name]
                target[j] = max(float(prev[j]) * 0.82, float(raw[i, j]) * 0.75)
            alpha = 0.65
        else:
            target = prev * 0.86
            alpha = 1.0
        out[i] = np.clip((1.0 - alpha) * prev + alpha * target, 0.0, 1.0)
        prev = out[i]
    return out


def fit_pose_pnp(points_px: np.ndarray, canonical_obj: np.ndarray,
                 camera_k: np.ndarray, prev_r: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    obj = canonical_obj[PNP_IDX].astype(np.float64).copy()
    obj -= 0.5 * (canonical_obj[MP_EAR_L] + canonical_obj[MP_EAR_R])
    img = np.asarray(points_px[PNP_IDX], dtype=np.float64)
    dist = np.zeros(4, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, camera_k, dist, flags=cv2.SOLVEPNP_EPNP)
    if ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, camera_k, dist, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not ok:
        return (prev_r.copy() if prev_r is not None else np.eye(3, dtype=np.float64)), False
    r_cv, _ = cv2.Rodrigues(rvec)
    r_renderer = np.diag([1.0, -1.0, -1.0]) @ r_cv
    if np.linalg.det(r_renderer) < 0:
        r_renderer[:, -1] *= -1.0
    return r_renderer.astype(np.float64), True


def euler_yxz(r: np.ndarray) -> Tuple[float, float, float]:
    e = Rotation.from_matrix(r).as_euler("YXZ", degrees=True)
    return float(e[0]), float(e[1]), float(e[2])


def slerp_one(r0: np.ndarray, r1: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 0.999:
        return r1
    if alpha <= 0.001:
        return r0
    rots = Rotation.from_matrix(np.stack([r0, r1], axis=0))
    s = Slerp([0.0, 1.0], rots)
    return s([float(alpha)]).as_matrix()[0]


def compute_pose_tracks(stream: MeshStream, fw: int, fh: int,
                        canonical: np.ndarray) -> PoseTracks:
    n = len(stream.frame)
    camera_k = np.asarray([[FOCAL_LEN, 0.0, fw / 2.0], [0.0, FOCAL_LEN, fh / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    center_raw = np.zeros((n, 2), dtype=np.float64)
    scale_raw = np.zeros(n, dtype=np.float64)
    center_s = np.zeros((n, 2), dtype=np.float64)
    scale_s = np.zeros(n, dtype=np.float64)
    rotations = np.zeros((n, 3, 3), dtype=np.float64)
    pose_ok = np.zeros(n, dtype=bool)

    prev_r: Optional[np.ndarray] = None
    prev_cls = ""
    prev_source = ""
    for i in range(n):
        c, sc = mesh_center_scale(stream.projected_px[i])
        r, ok = fit_pose_pnp(stream.projected_px[i], canonical, camera_k, prev_r)
        cls = source_class(str(stream.mesh_source[i]))
        source_changed = str(stream.mesh_source[i]) != prev_source

        if i == 0:
            center_s[i] = c
            scale_s[i] = sc
            rotations[i] = r
        else:
            cand_c = c.copy()
            if source_changed:
                delta = cand_c - center_s[i - 1]
                norm = float(np.linalg.norm(delta))
                max_step = max(0.045 * sc, 2.5)
                if norm > max_step:
                    cand_c = center_s[i - 1] + delta * (max_step / max(norm, 1e-6))
            alpha = {"observed": 0.72, "profile": 0.55, "interpolated": 0.35}[cls]
            if source_changed:
                alpha = min(alpha, 0.24)
            center_s[i] = (1.0 - alpha) * center_s[i - 1] + alpha * cand_c
            scale_s[i] = (1.0 - alpha) * scale_s[i - 1] + alpha * sc
            rotations[i] = slerp_one(rotations[i - 1], r, alpha)

        center_raw[i] = c
        scale_raw[i] = sc
        pose_ok[i] = ok
        prev_r = rotations[i]
        prev_cls = cls
        prev_source = str(stream.mesh_source[i])

    eulers = np.asarray([euler_yxz(rotations[i]) for i in range(n)], dtype=np.float64)
    return PoseTracks(
        center_raw=center_raw,
        scale_raw=scale_raw,
        center_smooth=center_s,
        scale_smooth=scale_s,
        rotations=rotations,
        yaw=eulers[:, 0],
        pitch=eulers[:, 1],
        roll=eulers[:, 2],
        pose_ok=pose_ok,
    )


def stream_transform_yaw(head_transform: np.ndarray) -> np.ndarray:
    out = []
    for t in head_transform:
        r = np.asarray(t[:3, :3], dtype=np.float64)
        s = max(float(np.linalg.norm(r[:, 0])), 1e-8)
        r = r / s
        if np.linalg.det(r) < 0:
            r[:, -1] *= -1.0
        out.append(euler_yxz(r)[0])
    return np.asarray(out, dtype=np.float64)


def control_dict(row: np.ndarray) -> Dict[str, float]:
    return {name: float(row[i]) for i, name in enumerate(DRIVER_TARGETS)}


def tint_rgba_by_source(rgba: np.ndarray, source: str) -> np.ndarray:
    cls = source_class(source)
    tint = SOURCE_TINT_RGB[cls]
    if tint is None:
        return rgba
    out = rgba.copy()
    alpha = out[:, :, 3] > 0
    rgb = out[:, :, :3].astype(np.float32) / 255.0
    strength = 0.32 if cls == "profile" else 0.42
    rgb[alpha] = (1.0 - strength) * rgb[alpha] + strength * tint[None, :]
    out[:, :, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return out


def alpha_bbox(rgba: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 10)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_bars(img: np.ndarray, controls: Dict[str, float]) -> None:
    h, w = img.shape[:2]
    rows = [
        ("jaw", controls.get("jawOpen", 0.0), (0, 220, 255)),
        ("sml", max(controls.get("mouthSmileLeft", 0.0), controls.get("mouthSmileRight", 0.0)), (0, 210, 90)),
        ("blk", max(controls.get("eyeBlinkLeft", 0.0), controls.get("eyeBlinkRight", 0.0)), (255, 120, 210)),
        ("brw", controls.get("browInnerUp", 0.0), (255, 180, 40)),
    ]
    x0 = max(8, w - 112)
    y0 = h - 96
    for k, (label, val, col) in enumerate(rows):
        y = y0 + k * 21
        cv2.rectangle(img, (x0, y), (x0 + 70, y + 10), (20, 20, 20), -1)
        cv2.rectangle(img, (x0, y), (x0 + int(70 * clip01(val)), y + 10), col, -1)
        draw_text(img, label, (x0 + 76, y + 10), col, 0.34, 1)


def draw_avatar_osd(img: np.ndarray, source: str, yaw: float, controls: Dict[str, float],
                    center: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> None:
    cls = source_class(source)
    col = SOURCE_COLORS_BGR[cls]
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 8)
    draw_text(img, f"mesh-driven avatar | {source} | yaw={yaw:+.0f}",
              (14, 34), col, 0.62, 1)
    draw_text(img, f"jaw={controls.get('jawOpen', 0.0):.2f} blink={max(controls.get('eyeBlinkLeft', 0.0), controls.get('eyeBlinkRight', 0.0)):.2f} smile={max(controls.get('mouthSmileLeft', 0.0), controls.get('mouthSmileRight', 0.0)):.2f}",
              (14, 60), col, 0.48, 1)
    cv2.circle(img, (int(round(center[0])), int(round(center[1]))), 6, (0, 255, 255), -1, cv2.LINE_AA)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
    draw_bars(img, controls)


def pearsonr_np(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3:
        return float("nan")
    x_arr = x_arr - np.mean(x_arr)
    y_arr = y_arr - np.mean(y_arr)
    den = float(np.linalg.norm(x_arr) * np.linalg.norm(y_arr))
    if den < 1e-12:
        return float("nan")
    return float(np.dot(x_arr, y_arr) / den)


def summarize(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(len(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95.0)),
        "max": float(np.max(v)),
    }


def boundary_pop(values_xy: np.ndarray, sources: np.ndarray, scale: np.ndarray) -> Dict:
    vals = []
    rows = []
    for i in range(1, len(sources)):
        if str(sources[i]) == str(sources[i - 1]):
            continue
        jump = float(np.linalg.norm(values_xy[i] - values_xy[i - 1]) / max(float(scale[i]), 1.0))
        vals.append(jump)
        rows.append({
            "frame": int(i),
            "from": str(sources[i - 1]),
            "to": str(sources[i]),
            "jump_over_head_scale": jump,
        })
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n_transitions": int(len(vals)),
        "mean_jump_over_head_scale": float(np.mean(arr)) if len(arr) else 0.0,
        "max_jump_over_head_scale": float(np.max(arr)) if len(arr) else 0.0,
        "transitions": rows,
        "pass_le_0p12": bool(len(arr) == 0 or float(np.max(arr)) <= 0.12),
    }


def encode_master_and_preview(fps: float) -> Tuple[float, float]:
    subprocess.run([
        "ffmpeg", "-y", "-i", RAW_PATH,
        "-vcodec", "libx264", "-crf", "22", "-preset", "fast",
        "-movflags", "+faststart", MASTER_PATH,
    ], check=True, capture_output=True)

    attempts = [
        ("scale=720:-2", "32"),
        ("scale=720:-2", "35"),
        ("scale=600:-2", "36"),
        ("scale=540:-2", "38"),
    ]
    for vf, crf in attempts:
        subprocess.run([
            "ffmpeg", "-y", "-i", RAW_PATH,
            "-vf", vf,
            "-vcodec", "libx264", "-crf", crf, "-preset", "fast",
            "-r", f"{fps:.6f}",
            "-movflags", "+faststart", PREVIEW_PATH,
        ], check=True, capture_output=True)
        if os.path.getsize(PREVIEW_PATH) < 8 * 1024 * 1024:
            break

    master_mb = os.path.getsize(MASTER_PATH) / 1024**2
    preview_mb = os.path.getsize(PREVIEW_PATH) / 1024**2
    if os.path.exists(RAW_PATH):
        os.remove(RAW_PATH)
    return master_mb, preview_mb


def build_montage(saved: Dict[str, Dict]) -> None:
    order = ["frontal", "three_quarter", "profile", "interpolated_back", "mouth_open", "close_up"]
    cells = []
    for label in order:
        row = saved.get(label)
        if row is None:
            cell = np.full((320, 540, 3), 30, dtype=np.uint8)
            title = f"{label}: missing"
        else:
            cell = cv2.resize(row["image"], (540, 320), interpolation=cv2.INTER_AREA)
            title = f"{label} f{row['frame']:04d} {row['source']}"
        draw_text(cell, title, (10, 24), (0, 255, 255), 0.48, 1)
        cells.append(cell)
    montage = np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])])
    cv2.imwrite(MONTAGE_PATH, montage)


def glb_morph_inventory(glb: EmojiGLB) -> Dict:
    targets = sorted(set(name for part in glb.parts for name in part["tnames"]))
    arkit_missing = [name for name in ARKIT_NAMES if name not in targets]
    essential_missing = [name for name in ESSENTIAL_TARGETS if name not in targets]
    return {
        "part_count": int(len(glb.parts)),
        "target_count": int(len(targets)),
        "targets": targets,
        "essential_targets": ESSENTIAL_TARGETS,
        "essential_missing": essential_missing,
        "arkit52_missing_from_glb": arkit_missing,
    }


def arkit_audit(stream: MeshStream, controls: np.ndarray) -> Dict:
    arkit_idx = {name: i for i, name in enumerate(stream.arkit_names)}
    rows = {}
    for name in DRIVER_TARGETS:
        if name not in arkit_idx:
            continue
        r = pearsonr_np(controls[:, DRIVER_TARGETS.index(name)], stream.arkit52[:, arkit_idx[name]])
        rows[name] = r
    selected = {k: rows.get(k) for k in ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight", "browInnerUp"]}
    vals = np.asarray([v for v in rows.values() if np.isfinite(v)], dtype=np.float64)
    return {
        "note": "arkit52_corrected was not used to drive the avatar; this is audit-only correlation.",
        "overlap_target_count": int(len(rows)),
        "mean_r": float(np.mean(vals)) if len(vals) else float("nan"),
        "selected_r": selected,
    }


def scan_script_for_blocked_terms() -> Dict:
    terms = [
        r"torch[.]cuda",
        "".join(["pytorch", "3d"]),
        "".join(["nvdi", "ffrast"]),
    ]
    hits = {}
    for term in terms:
        proc = subprocess.run(
            ["rg", "-n", term, os.path.basename(__file__)],
            cwd=OUT_DIR,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            hits[term] = proc.stdout.strip().splitlines()
        else:
            hits[term] = []
    return {"terms": terms, "hits": hits, "pass": all(len(v) == 0 for v in hits.values())}


def write_notes(report: Dict) -> None:
    m = report["metrics"]
    checks = report["success_checks"]
    lines = [
        "# Mesh Avatar Driver V1 Notes",
        "",
        "## Outputs",
        f"- Script: `{OUT_DIR}/mesh_avatar_driver_v1.py`",
        f"- 3-up master MP4: `{MASTER_PATH}`",
        f"- Preview MP4: `{PREVIEW_PATH}`",
        f"- Proof montage: `{MONTAGE_PATH}`",
        f"- JSON report: `{REPORT_PATH}`",
        f"- Control stream: `{CONTROL_STREAM_PATH}`",
        "",
        "## Driver Contract",
        "- Primary driver is `mesh_cascade_v4_stream.npz` geometry: `projected_px` / `verts`.",
        "- `arkit52_corrected` is audit-only and is not used to render the avatar.",
        "- `head_transform` is audit-only and is not used for primary pose, placement, or scale.",
        "- Pose comes from mesh-only PnP over canonical 468 landmarks.",
        "- Placement and scale come from mesh landmarks: eyes, ears, top/chin, and ear span.",
        "- Expressions come from mesh geometry ratios calibrated against high-confidence frontal observed frames.",
        "",
        "## GLB Morph Targets",
        f"- GLB target count: `{report['glb_morphs']['target_count']}`.",
        f"- Essential missing: `{report['glb_morphs']['essential_missing']}`.",
        f"- ARKit-52 missing from this GLB: `{report['glb_morphs']['arkit52_missing_from_glb']}`.",
        "",
        "## Success Metrics",
        f"- Alpha nonblank frames: `{m['alpha_coverage']['frames_with_alpha']}/{m['alpha_coverage']['total_frames']}` = `{m['alpha_coverage']['pct']:.2f}%`.",
        f"- Avatar center error/head median `{m['center_error_over_head_scale']['median']:.4f}`, p95 `{m['center_error_over_head_scale']['p95']:.4f}`.",
        f"- Avatar bbox scale ratio p95 `{m['bbox_scale_ratio']['p95']:.4f}`.",
        f"- Source-boundary center pop max `{m['source_boundary_avatar_center_pop']['max_jump_over_head_scale']:.4f}`.",
        f"- JawOpen vs mesh aperture r frontal |yaw|<30: `{m['mouth_self_consistency']['jaw_vs_aperture_r_frontal_abs_yaw_lt_30']:.4f}`.",
        f"- JawOpen vs mesh aperture r observed non-profile: `{m['mouth_self_consistency']['jaw_vs_aperture_r_observed_non_profile']:.4f}`.",
        f"- f485 jawOpen `{m['mouth_self_consistency']['frame_485']['jawOpen']:.4f}`, aperture `{m['mouth_self_consistency']['frame_485']['aperture']:.4f}`, source `{m['mouth_self_consistency']['frame_485']['source']}`.",
        "",
        "## Pass/Fail",
    ]
    for k, v in checks.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Honest Limits",
        "- Profile-fit frames are amber. Only mouth/jaw geometry is trusted there; other controls hold/decay from prior observed frames.",
        "- Interpolated/back-head frames are blue. Expression is held/decayed rather than invented.",
        "- The avatar is a generic GLB head. This proof demonstrates mesh-to-rig driving, not identity likeness.",
        f"- Neutral-vs-canonical residual probe: mean `{report['calibration']['identity_residual']['mean']:.4f}`, median `{report['calibration']['identity_residual']['median']:.4f}`, p95 `{report['calibration']['identity_residual']['p95']:.4f}` normalized by ear span.",
        "- Mouth/expression metrics are self-consistency checks between mesh-derived control and mesh aperture, not ground truth.",
        "",
    ]
    with open(NOTES_PATH, "w") as f:
        f.write("\n".join(lines))


def run() -> Dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f"{LOG_PREFIX} Loading mesh stream...")
    stream = load_stream()
    fps, fw, fh, video_n = video_meta()
    if video_n and video_n != len(stream.frame):
        print(f"{LOG_PREFIX} Warning: video frame count {video_n} != stream {len(stream.frame)}")
    print(f"{LOG_PREFIX} Video {fw}x{fh} @ {fps:.3f}fps, frames={len(stream.frame)}")

    canonical = np.asarray(trimesh.load(CANONICAL_OBJ, force="mesh", process=False).vertices, dtype=np.float64)
    if canonical.shape[0] != V_CANON:
        raise RuntimeError(f"Expected {V_CANON} canonical vertices, got {canonical.shape}")

    print(f"{LOG_PREFIX} Computing mesh-only pose tracks...")
    pose = compute_pose_tracks(stream, fw, fh, canonical)

    print(f"{LOG_PREFIX} Calibrating neutral expression from frontal observed mesh frames...")
    cal = calibrate(stream, pose.yaw, canonical)
    raw_controls, feature_rows = derive_raw_controls(stream, cal)
    controls = retarget_controls_by_source(raw_controls, stream.mesh_source)

    np.savez_compressed(
        CONTROL_STREAM_PATH,
        frame=stream.frame,
        controls=controls.astype(np.float32),
        raw_controls=raw_controls.astype(np.float32),
        target_names=np.asarray(DRIVER_TARGETS),
        center_px=pose.center_smooth.astype(np.float32),
        raw_center_px=pose.center_raw.astype(np.float32),
        head_scale_px=pose.scale_smooth.astype(np.float32),
        raw_head_scale_px=pose.scale_raw.astype(np.float32),
        yaw_deg=pose.yaw.astype(np.float32),
        pitch_deg=pose.pitch.astype(np.float32),
        roll_deg=pose.roll.astype(np.float32),
        mesh_source=stream.mesh_source,
        pipeline_version=np.asarray([PIPELINE_VERSION]),
        primary_driver=np.asarray(["mesh_cascade_v4_stream.projected_px_and_verts"]),
    )

    print(f"{LOG_PREFIX} Loading GLB and renderer...")
    glb = EmojiGLB(GLB_PATH)
    morphs = glb_morph_inventory(glb)
    renderer = AvatarRenderer(fw, fh, focal=FOCAL_LEN)
    writer = cv2.VideoWriter(
        RAW_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (PANEL_W * 3, PANEL_H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {RAW_PATH}")

    proof_targets = {
        50: "frontal",
        89: "three_quarter",
        188: "profile",
        437: "interpolated_back",
        485: "mouth_open",
        828: "close_up",
    }
    saved: Dict[str, Dict] = {}

    alpha_centers = np.full((len(stream.frame), 2), np.nan, dtype=np.float64)
    alpha_nonblank = np.zeros(len(stream.frame), dtype=bool)
    center_err = np.full(len(stream.frame), np.nan, dtype=np.float64)
    bbox_ratio = np.full(len(stream.frame), np.nan, dtype=np.float64)
    avatar_center_for_pop = pose.center_smooth.copy()

    cap = cv2.VideoCapture(VIDEO_PATH)
    print(f"{LOG_PREFIX} Rendering mesh-driven avatar video...")
    for i in range(len(stream.frame)):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        weights = control_dict(controls[i])
        scale = float(np.clip(AVATAR_SCALE_K * pose.scale_smooth[i], AVATAR_SCALE_MIN, AVATAR_SCALE_MAX))
        t_norm = build_T_from_screen_pos(
            pose.rotations[i],
            float(pose.center_smooth[i, 0]),
            float(pose.center_smooth[i, 1]),
            fw,
            fh,
            FOCAL_LEN,
            Z_REF,
        )
        rgba = renderer.render(glb, t_norm, weights, scale=scale)
        rgba = tint_rgba_by_source(rgba, str(stream.mesh_source[i]))
        bbox = alpha_bbox(rgba)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            alpha_nonblank[i] = True
            alpha_centers[i] = [(x0 + x1) * 0.5, (y0 + y1) * 0.5]
            avatar_center_for_pop[i] = alpha_centers[i]
            center_err[i] = float(np.linalg.norm(alpha_centers[i] - pose.center_smooth[i]) / max(float(pose.scale_smooth[i]), 1.0))
            bbox_ratio[i] = float(max(x1 - x0 + 1, y1 - y0 + 1) / max(float(pose.scale_smooth[i]), 1.0))

        avatar_panel = composite_rgba_bgr(frame_bgr, rgba)
        draw_avatar_osd(
            avatar_panel,
            str(stream.mesh_source[i]),
            float(pose.yaw[i]),
            weights,
            pose.center_smooth[i],
            bbox,
        )
        wire_panel = draw_wireframe(
            frame_bgr,
            stream.projected_px[i],
            stream.faces,
            str(stream.mesh_source[i]),
            float(pose.yaw[i]),
            float(weights.get("jawOpen", 0.0)),
        )
        real_panel = frame_bgr.copy()
        draw_text(real_panel, f"real frame f{i:04d}", (14, 34), (255, 255, 255), 0.62, 1)
        panel = panelize(real_panel, wire_panel, avatar_panel)
        writer.write(panel)

        if i in proof_targets:
            label = proof_targets[i]
            proof_path = f"{OUT_DIR}/mesh_avatar_driver_v1_proof_{label}_f{i:04d}.jpg"
            cv2.imwrite(proof_path, panel)
            saved[label] = {
                "frame": int(i),
                "source": str(stream.mesh_source[i]),
                "path": proof_path,
                "image": panel.copy(),
            }

        if i % 50 == 0:
            pct = 100.0 * float(alpha_nonblank[: i + 1].sum()) / float(i + 1)
            print(
                f"{LOG_PREFIX} f{i:04d}/{len(stream.frame)} "
                f"{stream.mesh_source[i]:<22s} yaw={pose.yaw[i]:+6.1f} "
                f"jaw={weights.get('jawOpen', 0.0):.2f} alpha={pct:.1f}%"
            )

    cap.release()
    writer.release()
    renderer.close()
    build_montage(saved)
    master_mb, preview_mb = encode_master_and_preview(fps)

    aperture = np.asarray([row["mouth_aperture"] for row in feature_rows], dtype=np.float64)
    jaw = controls[:, DRIVER_TARGETS.index("jawOpen")].astype(np.float64)
    observed_non_profile = np.asarray([
        source_class(str(s)) == "observed" for s in stream.mesh_source
    ], dtype=bool)
    frontal = observed_non_profile & (np.abs(pose.yaw) < 30.0)
    stream_yaw = stream_transform_yaw(stream.head_transform)
    yaw_delta = np.abs(pose.yaw - stream_yaw)
    source_counts = {
        str(s): int((stream.mesh_source == s).sum())
        for s in sorted(set(stream.mesh_source.tolist()))
    }

    driven_target_max = {
        name: float(np.max(np.abs(controls[:, j])))
        for j, name in enumerate(DRIVER_TARGETS)
    }
    driven_targets = [name for name, vmax in driven_target_max.items() if vmax > 0.01]
    source_pop = boundary_pop(avatar_center_for_pop, stream.mesh_source, pose.scale_smooth)

    metrics = {
        "stream_coverage": {
            "total_frames": int(len(stream.frame)),
            "verts_populated": int(np.isfinite(stream.verts).all(axis=(1, 2)).sum()),
            "projected_px_populated": int(np.isfinite(stream.projected_px).all(axis=(1, 2)).sum()),
            "coverage_pct": 100.0 * float(np.isfinite(stream.projected_px).all(axis=(1, 2)).sum()) / float(len(stream.frame)),
            "source_counts": source_counts,
        },
        "alpha_coverage": {
            "frames_with_alpha": int(alpha_nonblank.sum()),
            "total_frames": int(len(stream.frame)),
            "pct": 100.0 * float(alpha_nonblank.sum()) / float(len(stream.frame)),
        },
        "center_error_over_head_scale": summarize(center_err),
        "bbox_scale_ratio": summarize(bbox_ratio),
        "source_boundary_avatar_center_pop": source_pop,
        "mouth_self_consistency": {
            "note": "self-consistency only: mesh-derived jawOpen vs mesh aperture, not external ground truth.",
            "jaw_vs_aperture_r_frontal_abs_yaw_lt_30": pearsonr_np(jaw[frontal], aperture[frontal]),
            "jaw_vs_aperture_r_observed_non_profile": pearsonr_np(jaw[observed_non_profile], aperture[observed_non_profile]),
            "frontal_n": int(frontal.sum()),
            "observed_non_profile_n": int(observed_non_profile.sum()),
            "frame_485": {
                "jawOpen": float(jaw[485]) if len(jaw) > 485 else float("nan"),
                "aperture": float(aperture[485]) if len(aperture) > 485 else float("nan"),
                "source": str(stream.mesh_source[485]) if len(stream.mesh_source) > 485 else "missing",
                "yaw_deg": float(pose.yaw[485]) if len(pose.yaw) > 485 else float("nan"),
            },
        },
        "head_transform_ablation_comparator": {
            "note": "stream head_transform was not used by the primary renderer; this compares mesh-PnP yaw to stream yaw.",
            "median_abs_yaw_delta_deg": float(np.median(yaw_delta[np.isfinite(yaw_delta)])),
            "p95_abs_yaw_delta_deg": float(np.percentile(yaw_delta[np.isfinite(yaw_delta)], 95.0)),
        },
        "arkit52_audit_comparison": arkit_audit(stream, controls),
        "driven_targets": {
            "targets_with_nonzero_motion": driven_targets,
            "max_abs_by_target": driven_target_max,
        },
    }

    scan = scan_script_for_blocked_terms()
    success_checks = {
        "stream_version_is_mesh_cascade_v4": True,
        "avatar_rendered_alpha_ge_99pct": bool(metrics["alpha_coverage"]["pct"] >= 99.0),
        "stream_mesh_coverage_100pct": bool(metrics["stream_coverage"]["coverage_pct"] >= 100.0),
        "center_median_le_0p06": bool(metrics["center_error_over_head_scale"]["median"] <= 0.06),
        "center_p95_le_0p15": bool(metrics["center_error_over_head_scale"]["p95"] <= 0.15),
        "bbox_ratio_p95_in_0p70_1p35": bool(0.70 <= metrics["bbox_scale_ratio"]["p95"] <= 1.35),
        "source_boundary_pop_le_0p12": bool(source_pop["max_jump_over_head_scale"] <= 0.12),
        "jaw_r_frontal_ge_0p85": bool(metrics["mouth_self_consistency"]["jaw_vs_aperture_r_frontal_abs_yaw_lt_30"] >= 0.85),
        "jaw_r_observed_non_profile_ge_0p75": bool(metrics["mouth_self_consistency"]["jaw_vs_aperture_r_observed_non_profile"] >= 0.75),
        "f485_mouth_open_driven": bool(metrics["mouth_self_consistency"]["frame_485"]["jawOpen"] >= 0.25),
        "essential_glb_targets_present": bool(len(morphs["essential_missing"]) == 0),
        "blocked_render_dependencies_absent_in_script": bool(scan["pass"]),
    }
    kill_conditions = {
        "renderer_blank_offscreen_gt_2pct": bool(metrics["alpha_coverage"]["pct"] < 98.0),
        "essential_glb_morph_missing": bool(len(morphs["essential_missing"]) > 0),
        "center_p95_gt_0p20": bool(metrics["center_error_over_head_scale"]["p95"] > 0.20),
        "mouth_not_visibly_mesh_driven": bool(metrics["mouth_self_consistency"]["frame_485"]["jawOpen"] < 0.25),
        "source_boundary_pop_gt_0p12": bool(source_pop["max_jump_over_head_scale"] > 0.12),
        "blocked_render_dependency_hit": bool(not scan["pass"]),
    }
    kill_conditions["kill_hit"] = bool(any(v for k, v in kill_conditions.items() if k != "kill_hit"))

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_unix": time.time(),
        "primary_driver": "mesh_cascade_v4_stream.projected_px_and_verts",
        "audit_only_inputs": ["arkit52_corrected", "head_transform"],
        "paths": {
            "video": VIDEO_PATH,
            "stream": STREAM_PATH,
            "glb": GLB_PATH,
            "master": MASTER_PATH,
            "preview": PREVIEW_PATH,
            "montage": MONTAGE_PATH,
            "notes": NOTES_PATH,
            "report": REPORT_PATH,
            "control_stream": CONTROL_STREAM_PATH,
        },
        "output_sizes_mb": {
            "master": round(master_mb, 3),
            "preview": round(preview_mb, 3),
            "montage": round(os.path.getsize(MONTAGE_PATH) / 1024**2, 3) if os.path.exists(MONTAGE_PATH) else 0.0,
            "control_stream": round(os.path.getsize(CONTROL_STREAM_PATH) / 1024**2, 3),
        },
        "calibration": {
            "neutral_frame_count": cal.neutral_frame_count,
            "neutral_criteria": "observed source, mesh_conf>=0.95, abs(mesh_PnP_yaw)<8deg; fallback widens if needed",
            "feature_base": cal.feature_base,
            "feature_lo_observed_p02": cal.feature_lo,
            "feature_hi_observed_p98": cal.feature_hi,
            "identity_residual": cal.identity_residual,
        },
        "glb_morphs": morphs,
        "metrics": metrics,
        "proof_frames": {
            label: {k: v for k, v in row.items() if k != "image"}
            for label, row in saved.items()
        },
        "mps_no_disallowed_renderer_scan": scan,
        "success_checks": success_checks,
        "kill_conditions": kill_conditions,
        "honest_limits": [
            "Profile-fit frames are tinted amber; only mouth/jaw geometry is trusted there.",
            "Interpolated/back-head frames are tinted blue; expressions are held/decayed instead of invented.",
            "The GLB is a generic emoji head; this proof demonstrates mesh-to-rig control, not identity likeness.",
            "Mouth/expression correlations are self-consistency checks, not ground truth.",
        ],
        "processing_time_s": round(time.time() - t0, 2),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    write_notes(report)

    print(f"\n{LOG_PREFIX} Done")
    print(f"  Master:  {MASTER_PATH} ({master_mb:.2f} MB)")
    print(f"  Preview: {PREVIEW_PATH} ({preview_mb:.2f} MB)")
    print(f"  Montage: {MONTAGE_PATH}")
    print(f"  Report:  {REPORT_PATH}")
    print(f"  Notes:   {NOTES_PATH}")
    print(f"  Alpha:   {metrics['alpha_coverage']['pct']:.2f}%")
    print(f"  Center p95/head: {metrics['center_error_over_head_scale']['p95']:.4f}")
    print(f"  Jaw r frontal: {metrics['mouth_self_consistency']['jaw_vs_aperture_r_frontal_abs_yaw_lt_30']:.4f}")
    print(f"  Kill hit: {kill_conditions['kill_hit']}")
    return report


if __name__ == "__main__":
    run()
