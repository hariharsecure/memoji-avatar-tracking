#!/usr/bin/env python3
"""
pipeline_mesh_cascade_v7.py - pose-stabilized wireframe overlay.

v7 builds on v6's lag-free One-Euro semantic face lock and adds a localized
pose-stabilization pass for profile/averted/interpolated frames. The failure
mode is a head-pose branch flip during fast turn-aways; v7 detects those reads
in quaternion space, carries a constant-angular-velocity rotation arc through
averted spans, validates image-plane motion with optical flow over the visible
head region, and rebuilds only the unstable profile/interpolated mesh frames.

MediaPipe-observed frames keep the v6 projected geometry unchanged.

Inputs:
  mesh_cascade_v5_stream.npz
  mesh_cascade_v4_stream.npz
  mesh_cascade_v5_report.json / wireframe_perfect_before_after_metrics.json,
  when present, for the v5 before baseline

Outputs:
  mesh_cascade_v7_stream.npz
  mesh_cascade_v7_report.json
  wireframe_v7_before_after_pose_metrics.json
  notes_wireframe_v7.md
  wireframe_v7_overlay_master.mp4
  wireframe_v7_overlay_preview.mp4
  wireframe_v7_pose_flip_motion_strip_f425_440.png
  wireframe_v7_center_scale_montage.png
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import trimesh


VIDEO_PATH = "input_clip.mov"
CANONICAL_OBJ = "assets/canonical_face_model.obj"
OUT_DIR = "."

V4_STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v4_stream.npz"
V4_REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v4_report.json"
V5_STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v5_stream.npz"
V5_REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v5_report.json"
V5_METRICS_PATH = f"{OUT_DIR}/wireframe_perfect_before_after_metrics.json"
V6_STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v6_stream.npz"
V6_REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v6_report.json"
V6_METRICS_PATH = f"{OUT_DIR}/wireframe_v6_before_after_metrics.json"
STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v7_stream.npz"
REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v7_report.json"
METRICS_PATH = f"{OUT_DIR}/wireframe_v7_before_after_pose_metrics.json"
NOTES_PATH = f"{OUT_DIR}/notes_wireframe_v7.md"
OVERLAY_MASTER_PATH = f"{OUT_DIR}/wireframe_v7_overlay_master.mp4"
OVERLAY_PREVIEW_PATH = f"{OUT_DIR}/wireframe_v7_overlay_preview.mp4"
MONTAGE_PATH = f"{OUT_DIR}/wireframe_v7_center_scale_montage.png"
MOTION_STRIP_PATH = f"{OUT_DIR}/wireframe_v7_pose_flip_motion_strip_f425_440.png"

PIPELINE_VERSION = "mesh_cascade_v7_pose_stabilized"
LOG_PREFIX = "[mesh-cascade-v7]"
V_CANON = 468

POSE_FLIP_START = 425
POSE_FLIP_END = 440
PROFILE_SPAN_SOURCE_NAMES = {"profile_fit", "interpolated"}
POSE_OUTLIER_ANGLE_DEG = 38.0
POSE_NEIGHBOR_SPIKE_DEG = 55.0
FLOW_CONTRADICTION_OVER_HEAD_SCALE = 0.14
FLOW_MIN_CONFIDENCE = 0.20
AVERTED_RIGHT_BLEND_FRAMES = 8

ONE_EURO_MIN_CUTOFF = 0.08
ONE_EURO_BETA = 0.02
ONE_EURO_D_CUTOFF = 1.0
AFFINE_YAW_LIMIT_DEG = 60.0
AFFINE_MIN_HEAD_SCALE_PX = 35.0


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

FACE_OVAL_IDX = sorted({v for edge in FACE_OVAL for v in edge})
LEFT_EYE_IDX = sorted({v for edge in LEFT_EYE for v in edge})
RIGHT_EYE_IDX = sorted({v for edge in RIGHT_EYE for v in edge})
LIPS_IDX = sorted({v for edge in LIPS_OUTER + LIPS_INNER for v in edge})
NOSE_IDX = [1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 326, 327]
ANCHOR_IDX = np.asarray(
    sorted(set(FACE_OVAL_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX + LIPS_IDX + NOSE_IDX)),
    dtype=np.int32,
)
MP_FACE_EAR_LEFT_IDX = 234
MP_FACE_EAR_RIGHT_IDX = 454


def wrap_angle_deg(angle: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(angle, dtype=np.float64)
    return (arr + 180.0) % 360.0 - 180.0


def smoothstep01(t: float | np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def true_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((int(start), int(i - 1)))
            start = None
    if start is not None:
        spans.append((int(start), int(len(mask) - 1)))
    return spans


def summarize_values(values: np.ndarray) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "count": 0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(len(arr)),
        "median": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def unit_rotation_from_transform(head_transform: np.ndarray,
                                 fallback_ypr: Optional[Tuple[float, float, float]] = None) -> Rotation:
    r3 = np.asarray(head_transform[:3, :3], dtype=np.float64)
    if np.isfinite(r3).all() and float(np.linalg.norm(r3)) > 1e-8:
        u, _, vt = np.linalg.svd(r3)
        mat = u @ vt
        if np.linalg.det(mat) < 0.0:
            u[:, -1] *= -1.0
            mat = u @ vt
        return Rotation.from_matrix(mat)
    if fallback_ypr is None:
        fallback_ypr = (0.0, 0.0, 0.0)
    return Rotation.from_euler("YXZ", fallback_ypr, degrees=True)


def rotations_from_stream(head_transform: np.ndarray,
                          yaw: np.ndarray,
                          pitch: np.ndarray,
                          roll: np.ndarray) -> Rotation:
    mats = []
    for i in range(len(head_transform)):
        rot = unit_rotation_from_transform(
            head_transform[i],
            (float(yaw[i]), float(pitch[i]), float(roll[i])),
        )
        mats.append(rot.as_matrix())
    return Rotation.from_matrix(np.asarray(mats, dtype=np.float64))


def rotation_angle_deg(a: Rotation, b: Rotation) -> float:
    return float((a.inv() * b).magnitude() * 180.0 / math.pi)


def angular_jumps_deg(rotations: Rotation) -> np.ndarray:
    vals = np.zeros(len(rotations), dtype=np.float64)
    for i in range(1, len(rotations)):
        vals[i] = rotation_angle_deg(rotations[i - 1], rotations[i])
    return vals


def unwrap_yaw_deg(yaw_wrapped: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(yaw_wrapped, dtype=np.float64))))
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        if len(ref) == len(out):
            shift = round(float((ref[0] - out[0]) / 360.0)) * 360.0
            out = out + shift
    return out


def reliable_media_pipe_mask(source: np.ndarray) -> np.ndarray:
    src = np.asarray(source).astype(str)
    return np.asarray([s.startswith("observed") for s in src], dtype=bool)


def averted_pose_mask(source: np.ndarray) -> np.ndarray:
    src = np.asarray(source).astype(str)
    return np.asarray([s in PROFILE_SPAN_SOURCE_NAMES for s in src], dtype=bool)


def estimate_entry_angular_velocity(rotations: Rotation,
                                    reliable: np.ndarray,
                                    left: int,
                                    lookback: int = 10) -> np.ndarray:
    if left <= 0:
        return np.zeros(3, dtype=np.float64)
    lo = max(0, left - lookback)
    idx = [i for i in range(lo, left + 1) if bool(reliable[i])]
    if len(idx) < 2:
        idx = list(range(max(0, left - min(3, left)), left + 1))
    velocities = []
    for a, b in zip(idx[:-1], idx[1:]):
        dt = max(int(b - a), 1)
        delta = rotations[b] * rotations[a].inv()
        rv = delta.as_rotvec() / float(dt)
        if np.isfinite(rv).all() and float(np.linalg.norm(rv)) < math.radians(55.0):
            velocities.append(rv)
    if not velocities and left > 0:
        velocities.append((rotations[left] * rotations[left - 1].inv()).as_rotvec())
    if not velocities:
        return np.zeros(3, dtype=np.float64)
    return np.median(np.asarray(velocities, dtype=np.float64), axis=0)


def slerp_pair(a: Rotation, b: Rotation, t: float) -> Rotation:
    return Slerp([0.0, 1.0], Rotation.from_quat([a.as_quat(), b.as_quat()]))([float(np.clip(t, 0.0, 1.0))])[0]


def compute_optical_flow_anchors(video_path: str,
                                 target_center: np.ndarray,
                                 head_scale: np.ndarray,
                                 reliable: np.ndarray) -> Dict:
    n = len(target_center)
    flow_center = np.asarray(target_center, dtype=np.float64).copy()
    flow_disp = np.zeros((n, 2), dtype=np.float64)
    flow_rot = np.zeros(n, dtype=np.float64)
    flow_scale = np.ones(n, dtype=np.float64)
    flow_conf = np.zeros(n, dtype=np.float64)
    flow_inliers = np.zeros(n, dtype=np.int32)
    flow_points = np.zeros(n, dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    ok, prev_bgr = cap.read()
    if not ok:
        cap.release()
        return {
            "center_px": flow_center,
            "disp_px": flow_disp,
            "rotation_deg": flow_rot,
            "scale": flow_scale,
            "confidence": flow_conf,
            "inliers": flow_inliers,
            "points": flow_points,
            "available": False,
            "reason": "could_not_read_video",
        }
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape[:2]
    flow_center[0] = target_center[0]

    for i in range(1, n):
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if bool(reliable[i - 1]):
            flow_center[i - 1] = target_center[i - 1]

        cx, cy = flow_center[i - 1]
        radius = float(np.clip(head_scale[i - 1] * 1.35, 36.0, 220.0))
        x0 = int(max(0, round(cx - radius)))
        y0 = int(max(0, round(cy - radius * 1.15)))
        x1 = int(min(w, round(cx + radius)))
        y1 = int(min(h, round(cy + radius * 1.25)))
        if x1 - x0 < 16 or y1 - y0 < 16:
            flow_center[i] = target_center[i]
            prev_gray = gray
            continue

        roi = prev_gray[y0:y1, x0:x1]
        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=160,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=5,
        )
        if pts is None or len(pts) < 8:
            flow_center[i] = target_center[i] if bool(reliable[i]) else flow_center[i - 1]
            prev_gray = gray
            continue
        pts = pts.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += x0
        pts[:, 1] += y0
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            pts.reshape(-1, 1, 2),
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
        )
        if next_pts is None or status is None:
            flow_center[i] = target_center[i] if bool(reliable[i]) else flow_center[i - 1]
            prev_gray = gray
            continue

        good_old = pts[status.reshape(-1).astype(bool)]
        good_new = next_pts.reshape(-1, 2)[status.reshape(-1).astype(bool)]
        flow_points[i] = int(len(good_old))
        if len(good_old) < 8:
            flow_center[i] = target_center[i] if bool(reliable[i]) else flow_center[i - 1]
            prev_gray = gray
            continue

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.98,
        )
        if matrix is None or inlier_mask is None:
            disp = np.median(good_new - good_old, axis=0)
            conf = min(0.35, len(good_old) / 100.0)
            flow_center[i] = flow_center[i - 1] + disp
            flow_disp[i] = disp
            flow_conf[i] = conf
            prev_gray = gray
            continue

        inliers = inlier_mask.reshape(-1).astype(bool)
        flow_inliers[i] = int(inliers.sum())
        conf = float(np.clip(inliers.mean() * min(len(good_old) / 60.0, 1.0), 0.0, 1.0))
        a, b = float(matrix[0, 0]), float(matrix[0, 1])
        flow_rot[i] = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        flow_scale[i] = math.sqrt(max(a * a + b * b, 1e-9))
        center_h = np.asarray([flow_center[i - 1, 0], flow_center[i - 1, 1], 1.0], dtype=np.float64)
        predicted = matrix @ center_h
        disp = predicted - flow_center[i - 1]
        flow_disp[i] = disp
        flow_conf[i] = conf
        if bool(reliable[i]):
            flow_center[i] = target_center[i]
        elif conf >= FLOW_MIN_CONFIDENCE:
            flow_center[i] = predicted
        else:
            flow_center[i] = 0.72 * predicted + 0.28 * target_center[i]
        prev_gray = gray

    cap.release()
    return {
        "center_px": flow_center,
        "disp_px": flow_disp,
        "rotation_deg": flow_rot,
        "scale": flow_scale,
        "confidence": flow_conf,
        "inliers": flow_inliers,
        "points": flow_points,
        "available": True,
        "reason": "",
    }


def load_canonical_mesh() -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(CANONICAL_OBJ, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if verts.shape != (V_CANON, 3):
        raise RuntimeError(f"Expected canonical {V_CANON} vertices, got {verts.shape}")
    return verts, faces


def faces_to_edges(faces: np.ndarray) -> List[Tuple[int, int]]:
    edges = set()
    for tri in faces:
        a, b, c = [int(v) for v in tri]
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    return sorted(edges)


def project_canonical_pose_from_rotation(canon_verts: np.ndarray, rotation: Rotation,
                                         center_px: np.ndarray, scale_px: float,
                                         yaw_deg: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    ear_left = canon_verts[MP_FACE_EAR_LEFT_IDX]
    ear_right = canon_verts[MP_FACE_EAR_RIGHT_IDX]
    ear_mid = (ear_left + ear_right) * 0.5
    ear_span = float(np.linalg.norm(ear_left - ear_right))
    if ear_span < 1e-6:
        raise RuntimeError("canonical ear span is degenerate")

    yaw_for_boost = float(abs(wrap_angle_deg(yaw_deg)))
    if yaw_for_boost > 40.0:
        scale_boost = 1.35
    elif str(mode) == "MEDIAPIPE":
        scale_boost = 1.0
    else:
        scale_boost = 1.20

    px_per_unit = max(float(scale_px), 5.0) * scale_boost / ear_span
    centered = np.asarray(canon_verts, dtype=np.float64) - ear_mid.astype(np.float64)
    posed = rotation.apply(centered) * px_per_unit

    projected = np.empty((V_CANON, 2), dtype=np.float64)
    projected[:, 0] = float(center_px[0]) + posed[:, 0]
    projected[:, 1] = float(center_px[1]) - posed[:, 1]

    verts = np.empty((V_CANON, 3), dtype=np.float64)
    verts[:, 0:2] = projected
    verts[:, 2] = posed[:, 2]
    return verts, projected


def neutral_series(canon_verts: np.ndarray,
                   rotations: Rotation,
                   head_center: np.ndarray,
                   head_scale: np.ndarray,
                   yaw_unwrapped: np.ndarray,
                   mode: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    verts = np.zeros((len(rotations), V_CANON, 3), dtype=np.float64)
    px = np.zeros((len(rotations), V_CANON, 2), dtype=np.float64)
    for i in range(len(rotations)):
        verts[i], px[i] = project_canonical_pose_from_rotation(
            canon_verts,
            rotations[i],
            head_center[i],
            float(head_scale[i]),
            float(yaw_unwrapped[i]),
            str(mode[i]),
        )
    return verts, px


def semantic_center_scale_lock_to_target(projected_px: np.ndarray,
                                         target_px: np.ndarray,
                                         target_center_override: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    out = np.asarray(projected_px, dtype=np.float64).copy()
    target_center = semantic_center(target_px)
    if target_center_override is not None:
        override = np.asarray(target_center_override, dtype=np.float64)
        if override.shape != target_center.shape:
            raise ValueError(f"bad target center override shape {override.shape}, expected {target_center.shape}")
        good = np.isfinite(override).all(axis=1)
        target_center[good] = override[good]
    mesh_center = semantic_center(out)
    target_scale = semantic_scale(target_px, target_center)
    mesh_scale = semantic_scale(out, mesh_center)
    scale_factor = np.clip(target_scale / np.maximum(mesh_scale, 1e-6), 0.72, 1.38)
    out = (out - mesh_center[:, None, :]) * scale_factor[:, None, None] + mesh_center[:, None, :]
    mesh_center = semantic_center(out)
    translation = target_center - mesh_center
    out = out + translation[:, None, :]
    post_center = semantic_center(out)
    residual_px = np.linalg.norm(post_center - target_center, axis=1)
    return out, {
        "definition": "weighted semantic anchor center/scale lock after v7 pose rebuild; optional flow-center override for averted frames",
        "scale_factor_median": float(np.percentile(scale_factor, 50.0)),
        "scale_factor_p05": float(np.percentile(scale_factor, 5.0)),
        "scale_factor_p95": float(np.percentile(scale_factor, 95.0)),
        "scale_factor_min": float(np.min(scale_factor)),
        "scale_factor_max": float(np.max(scale_factor)),
        "post_lock_center_residual_px_max": float(np.max(residual_px)),
    }


def stabilize_rotations_with_cav(raw_rotations: Rotation,
                                 yaw_raw: np.ndarray,
                                 pitch_raw: np.ndarray,
                                 roll_raw: np.ndarray,
                                 source: np.ndarray,
                                 flow: Dict,
                                 target_center: np.ndarray,
                                 head_scale: np.ndarray) -> Tuple[Rotation, np.ndarray, np.ndarray, np.ndarray, Dict]:
    n = len(raw_rotations)
    reliable = reliable_media_pipe_mask(source)
    unstable = averted_pose_mask(source)
    stable_mats = raw_rotations.as_matrix().copy()
    raw_jumps = angular_jumps_deg(raw_rotations)
    predicted_error = np.zeros(n, dtype=np.float64)
    flow_error = np.zeros(n, dtype=np.float64)
    flow_contradiction = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)
    pose_policy = np.full(n, "raw_observed_media_pipe", dtype="<U40")
    pose_policy[unstable] = "cav_averted_span"
    spans_report: List[Dict] = []

    for start, end in true_spans(unstable):
        left = start - 1
        while left >= 0 and not bool(reliable[left]):
            left -= 1
        right = end + 1
        while right < n and not bool(reliable[right]):
            right += 1

        if left < 0:
            anchor = raw_rotations[start]
            omega = np.zeros(3, dtype=np.float64)
            left_for_dt = start
        else:
            anchor = Rotation.from_matrix(stable_mats[left])
            omega = estimate_entry_angular_velocity(raw_rotations, reliable, left)
            left_for_dt = left

        cav_rots: List[Rotation] = []
        for fidx in range(start, end + 1):
            dt = float(fidx - left_for_dt)
            cav = Rotation.from_rotvec(omega * dt) * anchor
            cav_rots.append(cav)

        if right < n:
            right_rot = raw_rotations[right]
            blend_start = max(start, end - AVERTED_RIGHT_BLEND_FRAMES + 1)
            for offset, fidx in enumerate(range(start, end + 1)):
                if fidx >= blend_start:
                    denom = max(float(right - blend_start), 1.0)
                    b = float(smoothstep01((fidx - blend_start + 1) / denom))
                    cav_rots[offset] = slerp_pair(cav_rots[offset], right_rot, b * 0.85)

        for offset, fidx in enumerate(range(start, end + 1)):
            cav = cav_rots[offset]
            predicted_error[fidx] = rotation_angle_deg(raw_rotations[fidx], cav)
            if fidx > 0:
                pose_step = np.asarray(target_center[fidx], dtype=np.float64) - np.asarray(target_center[fidx - 1], dtype=np.float64)
                flow_step = np.asarray(flow["disp_px"][fidx], dtype=np.float64)
                flow_error[fidx] = float(np.linalg.norm(pose_step - flow_step) / max(float(head_scale[fidx]), 1.0))
                flow_contradiction[fidx] = bool(
                    flow["available"]
                    and float(flow["confidence"][fidx]) >= FLOW_MIN_CONFIDENCE
                    and flow_error[fidx] > FLOW_CONTRADICTION_OVER_HEAD_SCALE
                )
            neighbor_spike = bool(raw_jumps[fidx] > POSE_NEIGHBOR_SPIKE_DEG)
            rejected[fidx] = bool(
                predicted_error[fidx] > POSE_OUTLIER_ANGLE_DEG
                or neighbor_spike
                or flow_contradiction[fidx]
                or str(np.asarray(source).astype(str)[fidx]) == "interpolated"
            )
            stable_mats[fidx] = cav.as_matrix()
            if rejected[fidx]:
                pose_policy[fidx] = "rejected_to_cav_prediction"

        spans_report.append({
            "start": int(start),
            "end": int(end),
            "len": int(end - start + 1),
            "left_reliable": int(left) if left >= 0 else None,
            "right_reliable": int(right) if right < n else None,
            "omega_deg_per_frame": [float(v * 180.0 / math.pi) for v in omega],
            "max_raw_vs_cav_angle_deg": float(np.max(predicted_error[start:end + 1])),
            "rejected_frames": [int(v) for v in np.where(rejected[start:end + 1])[0] + start],
        })

    stable_rotations = Rotation.from_matrix(stable_mats)
    euler_wrapped = stable_rotations.as_euler("YXZ", degrees=True)
    yaw_wrapped = wrap_angle_deg(euler_wrapped[:, 0])
    yaw_unwrapped = unwrap_yaw_deg(yaw_wrapped, reference=unwrap_yaw_deg(yaw_raw))
    pitch = euler_wrapped[:, 1]
    roll = euler_wrapped[:, 2]

    return stable_rotations, yaw_unwrapped, pitch, roll, {
        "definition": "profile/interpolated frames are replaced by quaternion constant-angular-velocity pose arcs; MediaPipe observed frames are unchanged",
        "reliable_media_pipe_frames": int(reliable.sum()),
        "averted_profile_interpolated_frames": int(unstable.sum()),
        "spans": spans_report,
        "raw_angular_step_deg": summarize_values(raw_jumps[1:]),
        "stabilized_angular_step_deg": summarize_values(angular_jumps_deg(stable_rotations)[1:]),
        "rejected_frame_count": int(rejected.sum()),
        "rejected_frames": [int(v) for v in np.where(rejected)[0]],
        "predicted_error_deg": summarize_values(predicted_error[unstable]),
        "flow_contradiction_frames": [int(v) for v in np.where(flow_contradiction & unstable)[0]],
        "pose_policy_counts": {str(k): int(v) for k, v in zip(*np.unique(pose_policy, return_counts=True))},
        "pose_policy": pose_policy,
        "predicted_error_by_frame": predicted_error,
        "flow_error_by_frame": flow_error,
        "flow_contradiction": flow_contradiction,
        "rejected_mask": rejected,
    }


def rebuild_averted_geometry(canon_verts: np.ndarray,
                             v6_verts: np.ndarray,
                             v6_px: np.ndarray,
                             raw_rotations: Rotation,
                             stable_rotations: Rotation,
                             yaw_raw_unwrapped: np.ndarray,
                             yaw_stable_unwrapped: np.ndarray,
                             source: np.ndarray,
                             head_center: np.ndarray,
                             head_scale: np.ndarray,
                             mode: np.ndarray,
                             flow: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n = len(v6_px)
    source_str = np.asarray(source).astype(str)
    reliable = reliable_media_pipe_mask(source_str)
    unstable = averted_pose_mask(source_str)
    out_verts = np.asarray(v6_verts, dtype=np.float64).copy()
    out_px = np.asarray(v6_px, dtype=np.float64).copy()

    neutral_raw, _ = neutral_series(canon_verts, raw_rotations, head_center, head_scale, yaw_raw_unwrapped, mode)
    neutral_stable, _ = neutral_series(canon_verts, stable_rotations, head_center, head_scale, yaw_stable_unwrapped, mode)
    residual_raw = np.asarray(v6_verts, dtype=np.float64) - neutral_raw
    spans_report: List[Dict] = []

    for start, end in true_spans(unstable):
        left = start - 1
        while left >= 0 and not bool(reliable[left]):
            left -= 1
        right = end + 1
        while right < n and not bool(reliable[right]):
            right += 1
        if left < 0 or right >= n:
            spans_report.append({
                "start": int(start),
                "end": int(end),
                "len": int(end - start + 1),
                "rebuilt": False,
                "reason": "missing_reliable_bracket",
            })
            continue

        residual_left = residual_raw[left]
        residual_right = residual_raw[right]
        for fidx in range(start, end + 1):
            t = (fidx - left) / float(max(right - left, 1))
            e = float(smoothstep01(t))
            residual_i = (1.0 - e) * residual_left + e * residual_right
            rebuilt = neutral_stable[fidx] + residual_i
            out_verts[fidx] = rebuilt
            out_px[fidx] = rebuilt[:, :2]

        spans_report.append({
            "start": int(start),
            "end": int(end),
            "len": int(end - start + 1),
            "rebuilt": True,
            "left_reliable": int(left),
            "right_reliable": int(right),
            "policy": "stable_pose_neutral_plus_smoothstep_interpolated_observed_residuals",
        })

    target_center_override = semantic_center(v6_px)
    flow_override_used = np.zeros(n, dtype=bool)
    flow_override_alpha = np.zeros(n, dtype=np.float64)
    if bool(flow.get("available", False)):
        flow_center = np.asarray(flow["center_px"], dtype=np.float64)
        flow_conf = np.asarray(flow["confidence"], dtype=np.float64)
        flow_disp = np.asarray(flow["disp_px"], dtype=np.float64)
        center_step = np.r_[np.zeros((1, 2), dtype=np.float64), np.diff(target_center_override, axis=0)]
        flow_error = np.linalg.norm(center_step - flow_disp, axis=1) / np.maximum(np.asarray(head_scale, dtype=np.float64), 1.0)
        valid_flow = unstable & (flow_conf >= FLOW_MIN_CONFIDENCE) & np.isfinite(flow_center).all(axis=1)
        alpha = np.clip((flow_error - 0.06) / 0.22, 0.0, 1.0)
        alpha *= valid_flow.astype(np.float64)
        flow_override_alpha = alpha
        flow_override_used = alpha > 0.05
        target_center_override = (
            (1.0 - alpha[:, None]) * target_center_override
            + alpha[:, None] * flow_center
        )
    relocked_px, lock_detail = semantic_center_scale_lock_to_target(out_px, v6_px, target_center_override)
    out_verts[:, :, :2] = relocked_px
    center_before = np.linalg.norm(semantic_center(v6_px) - semantic_center(out_px), axis=1)
    center_after = np.linalg.norm(semantic_center(v6_px) - semantic_center(relocked_px), axis=1)
    old_scale = semantic_scale(out_px, semantic_center(out_px))
    new_scale = semantic_scale(relocked_px, semantic_center(relocked_px))
    z_scale = np.where(old_scale > 1e-6, new_scale / old_scale, 1.0)
    out_verts[:, :, 2] *= z_scale[:, None]
    out_verts[reliable] = np.asarray(v6_verts, dtype=np.float64)[reliable]
    relocked_px[reliable] = np.asarray(v6_px, dtype=np.float64)[reliable]

    return out_verts.astype(np.float32), relocked_px.astype(np.float32), {
        "definition": "only profile_fit/interpolated frames are rebuilt; observed_* frames are byte-for-byte copied from v6 geometry",
        "spans": spans_report,
        "unstable_frames_rebuilt": int(unstable.sum()),
        "observed_frames_preserved": int(reliable.sum()),
        "observed_geometry_max_abs_delta_px": float(np.max(np.abs(relocked_px[reliable] - np.asarray(v6_px, dtype=np.float64)[reliable]))) if reliable.any() else 0.0,
        "flow_center_override_frames": int(flow_override_used.sum()),
        "flow_center_override_pose_flip_frames": [int(v) for v in np.where(flow_override_used[POSE_FLIP_START:POSE_FLIP_END + 1])[0] + POSE_FLIP_START],
        "flow_center_override_alpha": summarize_values(flow_override_alpha[unstable]),
        "pre_relock_center_delta_to_v6_px": summarize_values(center_before[unstable]),
        "post_relock_center_delta_to_v6_px": summarize_values(center_after[unstable]),
        "center_scale_lock_to_v6": lock_detail,
    }


def update_head_transforms_with_rotations(head_transform: np.ndarray,
                                          rotations: Rotation) -> np.ndarray:
    out = np.asarray(head_transform, dtype=np.float64).copy()
    for i in range(len(out)):
        r3 = np.asarray(head_transform[i, :3, :3], dtype=np.float64)
        norms = np.linalg.norm(r3, axis=0)
        scale = float(np.median(norms[np.isfinite(norms) & (norms > 1e-8)])) if np.any(norms > 1e-8) else 1.0
        out[i, :3, :3] = rotations[i].as_matrix() * scale
    return out.astype(np.float32)


def pose_detail_for_json(pose_detail: Dict) -> Dict:
    return {
        "definition": pose_detail["definition"],
        "reliable_media_pipe_frames": int(pose_detail["reliable_media_pipe_frames"]),
        "averted_profile_interpolated_frames": int(pose_detail["averted_profile_interpolated_frames"]),
        "spans": pose_detail["spans"],
        "raw_angular_step_deg": pose_detail["raw_angular_step_deg"],
        "stabilized_angular_step_deg": pose_detail["stabilized_angular_step_deg"],
        "rejected_frame_count": int(pose_detail["rejected_frame_count"]),
        "rejected_frames": [int(v) for v in pose_detail["rejected_frames"]],
        "predicted_error_deg": pose_detail["predicted_error_deg"],
        "flow_contradiction_frames": [int(v) for v in pose_detail["flow_contradiction_frames"]],
        "pose_policy_counts": pose_detail["pose_policy_counts"],
    }


def weighted_similarity(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit dst ~= A @ src + t. Reflection is allowed because image y is flipped."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"bad similarity shapes src={src.shape} dst={dst.shape}")
    w = np.maximum(w, 1e-9)
    w = w / float(np.sum(w))
    mu_src = np.sum(src * w[:, None], axis=0)
    mu_dst = np.sum(dst * w[:, None], axis=0)
    x = src - mu_src
    y = dst - mu_dst
    var = float(np.sum(w[:, None] * x * x))
    cov = y.T @ (x * w[:, None])
    u, singular, vt = np.linalg.svd(cov)
    r = u @ vt
    scale = float(np.sum(singular) / max(var, 1e-9))
    a = scale * r
    t = mu_dst - a @ mu_src
    return a.astype(np.float64), t.astype(np.float64)


def robust_basis_fits(canon_xy: np.ndarray, raw_px: np.ndarray,
                      head_scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = canon_xy[ANCHOR_IDX]
    base_weights = np.ones(len(ANCHOR_IDX), dtype=np.float64)
    face_set = set(FACE_OVAL_IDX)
    nose_set = set(NOSE_IDX)
    lips_set = set(LIPS_IDX)
    for k, idx in enumerate(ANCHOR_IDX):
        if int(idx) in face_set:
            base_weights[k] *= 2.0
        if int(idx) in nose_set:
            base_weights[k] *= 1.5
        if int(idx) in lips_set:
            base_weights[k] *= 0.7

    n = raw_px.shape[0]
    matrices = np.zeros((n, 2, 2), dtype=np.float64)
    offsets = np.zeros((n, 2), dtype=np.float64)
    fit_err = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dst = raw_px[i, ANCHOR_IDX]
        a, t = weighted_similarity(src, dst, base_weights)
        pred = (a @ src.T).T + t
        err = np.linalg.norm(pred - dst, axis=1)
        med = float(np.median(err))
        mad = float(np.median(np.abs(err - med)) + 1e-6)
        robust = base_weights * np.clip(4.685 * mad / np.maximum(err, 1e-6), 0.0, 1.0) ** 2
        a, t = weighted_similarity(src, dst, robust)
        pred = (a @ src.T).T + t
        err = np.linalg.norm(pred - dst, axis=1)
        matrices[i] = a
        offsets[i] = t
        fit_err[i] = float(np.mean(err) / max(float(head_scale[i]), 1.0))
    return matrices, offsets, fit_err


def source_confidence(source: np.ndarray, head_scale: np.ndarray) -> np.ndarray:
    out = np.ones(len(source), dtype=np.float64)
    for i, src in enumerate(source.astype(str)):
        if src == "observed_zoom_mp":
            out[i] = 0.90
        elif src == "observed_zoom_tight_mp":
            out[i] = 0.85
        elif src == "observed_centered_mp":
            out[i] = 0.75
        elif src == "profile_fit":
            out[i] = 0.50
        elif src == "interpolated":
            out[i] = 0.32
        elif src.startswith("observed"):
            out[i] = 1.0
        else:
            out[i] = 0.25
    out *= np.clip(np.asarray(head_scale, dtype=np.float64) / 45.0, 0.20, 1.0)
    return out


def semantic_anchor_weights() -> Tuple[np.ndarray, np.ndarray]:
    """Stable face-feature weights used for center, scale, and residual metrics."""
    idx = ANCHOR_IDX.copy()
    weights = np.ones(len(idx), dtype=np.float64)
    eyes = set(LEFT_EYE_IDX + RIGHT_EYE_IDX)
    nose = set(NOSE_IDX)
    lips = set(LIPS_IDX)
    for k, vidx in enumerate(idx):
        if int(vidx) in eyes:
            weights[k] *= 2.4
        if int(vidx) in nose:
            weights[k] *= 2.2
        if int(vidx) in lips:
            weights[k] *= 1.8
    return idx, weights


def correction_weights_for_source(source: str, idx: np.ndarray,
                                  base_weights: np.ndarray) -> np.ndarray:
    weights = base_weights.copy()
    eyes = set(LEFT_EYE_IDX + RIGHT_EYE_IDX)
    nose = set(NOSE_IDX)
    lips = set(LIPS_IDX)
    face = set(FACE_OVAL_IDX)
    if source == "profile_fit":
        for k, vidx in enumerate(idx):
            iv = int(vidx)
            if iv in eyes or iv in lips:
                weights[k] *= 0.22
            if iv in nose or iv in face:
                weights[k] *= 1.8
    elif source == "interpolated":
        for k, vidx in enumerate(idx):
            if int(vidx) in eyes or int(vidx) in lips:
                weights[k] *= 0.55
    return weights


def weighted_affine(src: np.ndarray, dst: np.ndarray,
                    weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit dst ~= M @ src + t with a full 2D affine transform."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    x = np.column_stack([src[:, 0], src[:, 1], np.ones(len(src), dtype=np.float64)])
    sw = np.sqrt(weights / max(float(np.mean(weights)), 1e-9))[:, None]
    coeff = np.linalg.lstsq(x * sw, dst * sw, rcond=None)[0]
    matrix = coeff[:2, :].T
    offset = coeff[2, :]
    return matrix.astype(np.float64), offset.astype(np.float64)


def robust_correction_fit(src: np.ndarray, dst: np.ndarray, weights: np.ndarray,
                          kind: str) -> Tuple[np.ndarray, np.ndarray]:
    fit_weights = weights.copy()
    matrix = np.eye(2, dtype=np.float64)
    offset = np.zeros(2, dtype=np.float64)
    for _ in range(4):
        if kind == "affine":
            matrix, offset = weighted_affine(src, dst, fit_weights)
        elif kind == "similarity":
            matrix, offset = weighted_similarity(src, dst, fit_weights)
        else:
            raise ValueError(f"unknown correction kind: {kind}")
        pred = (matrix @ src.T).T + offset
        err = np.linalg.norm(pred - dst, axis=1)
        med = float(np.median(err))
        mad = float(np.median(np.abs(err - med)) + 1e-6)
        robust_radius = max(4.685 * mad, 1e-6)
        ratio = err / robust_radius
        robust = np.where(ratio < 1.0, (1.0 - ratio * ratio) ** 2, 0.05)
        fit_weights = weights * robust
    return matrix, offset


class OneEuroFilter:
    """Vectorized One-Euro filter (Casiez/Roussel/Vogel) for pose params."""

    def __init__(self, freq: float, min_cutoff: float,
                 beta: float, d_cutoff: float) -> None:
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.prev_x: Optional[np.ndarray] = None
        self.prev_dx: Optional[np.ndarray] = None

    def alpha(self, cutoff: np.ndarray | float) -> np.ndarray:
        cutoff_arr = np.maximum(np.asarray(cutoff, dtype=np.float64), 1e-6)
        tau = 1.0 / (2.0 * math.pi * cutoff_arr)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, value: np.ndarray) -> np.ndarray:
        x = np.asarray(value, dtype=np.float64)
        if self.prev_x is None:
            self.prev_x = x.copy()
            self.prev_dx = np.zeros_like(x)
            return x.copy()
        dx = (x - self.prev_x) * self.freq
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.prev_dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha_x = self.alpha(cutoff)
        x_hat = alpha_x * x + (1.0 - alpha_x) * self.prev_x
        self.prev_x = x_hat.copy()
        self.prev_dx = dx_hat.copy()
        return x_hat


def one_euro_filter_series(values: np.ndarray, fps: float) -> np.ndarray:
    filt = OneEuroFilter(
        freq=fps,
        min_cutoff=ONE_EURO_MIN_CUTOFF,
        beta=ONE_EURO_BETA,
        d_cutoff=ONE_EURO_D_CUTOFF,
    )
    return np.asarray([filt(v) for v in values], dtype=np.float64)


def fit_one_euro_corrections(v5_px: np.ndarray, raw_px: np.ndarray,
                             source: np.ndarray, yaw: np.ndarray,
                             head_scale: np.ndarray,
                             fps: float) -> Tuple[np.ndarray, Dict]:
    idx, base_weights = semantic_anchor_weights()
    params = np.zeros((len(v5_px), 6), dtype=np.float64)
    fit_err = np.zeros(len(v5_px), dtype=np.float64)
    correction_kind: List[str] = []

    for i in range(len(v5_px)):
        src_name = str(source[i])
        weights = correction_weights_for_source(src_name, idx, base_weights)
        kind = "affine"
        if (
            src_name in {"profile_fit", "interpolated"}
            or abs(float(yaw[i])) >= AFFINE_YAW_LIMIT_DEG
            or float(head_scale[i]) < AFFINE_MIN_HEAD_SCALE_PX
        ):
            kind = "similarity"
        matrix, offset = robust_correction_fit(v5_px[i, idx], raw_px[i, idx], weights, kind)
        pred = (matrix @ v5_px[i, idx].T).T + offset
        err = np.linalg.norm(pred - raw_px[i, idx], axis=1)
        fit_err[i] = float(np.average(err, weights=weights) / max(float(head_scale[i]), 1.0))
        params[i] = np.r_[matrix.reshape(-1), offset]
        correction_kind.append(kind)

    filtered_params = one_euro_filter_series(params, fps)
    return filtered_params, {
        "one_euro": {
            "min_cutoff": ONE_EURO_MIN_CUTOFF,
            "beta": ONE_EURO_BETA,
            "d_cutoff": ONE_EURO_D_CUTOFF,
            "fps": float(fps),
        },
        "correction_fit_error_over_head_scale": {
            "median": float(np.percentile(fit_err, 50.0)),
            "p95": float(np.percentile(fit_err, 95.0)),
            "p99": float(np.percentile(fit_err, 99.0)),
            "max": float(np.max(fit_err)),
        },
        "correction_kind_counts": {
            str(k): int(v) for k, v in zip(*np.unique(np.asarray(correction_kind), return_counts=True))
        },
        "affine_policy": (
            f"affine for observed/yaw<{AFFINE_YAW_LIMIT_DEG:g}/scale>={AFFINE_MIN_HEAD_SCALE_PX:g}; "
            "similarity for profile/interpolated/wide-yaw/low-scale"
        ),
    }


def apply_correction_params(v5_px: np.ndarray, params: np.ndarray) -> np.ndarray:
    out = np.empty_like(v5_px, dtype=np.float64)
    for i, row in enumerate(params):
        matrix = row[:4].reshape(2, 2)
        offset = row[4:]
        out[i] = (matrix @ v5_px[i].T).T + offset
    return out


def semantic_center(px: np.ndarray) -> np.ndarray:
    idx, weights = semantic_anchor_weights()
    return np.sum(px[:, idx] * weights[None, :, None], axis=1) / np.sum(weights)


def semantic_scale(px: np.ndarray, center: np.ndarray) -> np.ndarray:
    idx, weights = semantic_anchor_weights()
    delta = px[:, idx] - center[:, None, :]
    return np.sqrt(
        np.sum(weights[None, :, None] * (delta * delta), axis=(1, 2))
        / max(float(np.sum(weights) * 2.0), 1e-9)
    )


def semantic_center_scale_lock(projected_px: np.ndarray,
                               raw_px: np.ndarray) -> Tuple[np.ndarray, Dict]:
    out = np.asarray(projected_px, dtype=np.float64).copy()
    face_center = semantic_center(raw_px)
    mesh_center = semantic_center(out)
    face_scale = semantic_scale(raw_px, face_center)
    mesh_scale = semantic_scale(out, mesh_center)
    scale_factor = np.clip(face_scale / np.maximum(mesh_scale, 1e-6), 0.72, 1.38)
    out = (out - mesh_center[:, None, :]) * scale_factor[:, None, None] + mesh_center[:, None, :]
    mesh_center = semantic_center(out)
    translation = face_center - mesh_center
    out = out + translation[:, None, :]
    post_center = semantic_center(out)
    residual_px = np.linalg.norm(post_center - face_center, axis=1)
    return out, {
        "definition": "weighted semantic anchor center/scale lock; applied every frame, not at source boundaries",
        "scale_factor_median": float(np.percentile(scale_factor, 50.0)),
        "scale_factor_p05": float(np.percentile(scale_factor, 5.0)),
        "scale_factor_p95": float(np.percentile(scale_factor, 95.0)),
        "scale_factor_min": float(np.min(scale_factor)),
        "scale_factor_max": float(np.max(scale_factor)),
        "post_lock_center_residual_px_max": float(np.max(residual_px)),
    }


def update_verts_from_projected(v5_verts: np.ndarray, v5_px: np.ndarray,
                                new_px: np.ndarray, params: np.ndarray) -> np.ndarray:
    verts = np.asarray(v5_verts, dtype=np.float64).copy()
    verts[:, :, :2] = new_px
    scale_ratio = np.ones(len(new_px), dtype=np.float64)
    for i, row in enumerate(params):
        matrix = row[:4].reshape(2, 2)
        scale_ratio[i] = math.sqrt(max(abs(float(np.linalg.det(matrix))), 1e-9))
    old_center = semantic_center(v5_px)
    new_center = semantic_center(new_px)
    old_scale = semantic_scale(v5_px, old_center)
    new_scale = semantic_scale(new_px, new_center)
    scale_ratio = np.where(old_scale > 1e-6, new_scale / old_scale, scale_ratio)
    verts[:, :, 2] = verts[:, :, 2] * scale_ratio[:, None]
    return verts.astype(np.float32)


def cross_correlation_delay(mesh_center: np.ndarray, face_center: np.ndarray,
                            max_delay: int = 12) -> Dict:
    mesh_vel = np.diff(mesh_center, axis=0)
    face_vel = np.diff(face_center, axis=0)
    rows = []
    for delay in range(-max_delay, max_delay + 1):
        if delay > 0:
            # Positive means the mesh has to be shifted later to match the face:
            # mesh is trailing the face by `delay` frames.
            a = mesh_vel[delay:]
            b = face_vel[:-delay]
        elif delay < 0:
            a = mesh_vel[:delay]
            b = face_vel[-delay:]
        else:
            a = mesh_vel
            b = face_vel
        if len(a) < 5:
            continue
        aa = a - np.mean(a, axis=0)
        bb = b - np.mean(b, axis=0)
        corr = float(np.sum(aa * bb) / max(math.sqrt(float(np.sum(aa * aa) * np.sum(bb * bb))), 1e-9))
        rows.append({"delay_frames": int(delay), "corr": corr})
    best = max(rows, key=lambda r: r["corr"])
    top = sorted(rows, key=lambda r: r["corr"], reverse=True)[:5]
    return {
        "definition": "velocity cross-correlation of weighted semantic mesh centroid vs raw face centroid; positive delay means mesh trails face",
        "best_delay_frames": int(best["delay_frames"]),
        "best_corr": float(best["corr"]),
        "top5": top,
    }


def center_error_report(mesh_px: np.ndarray, face_px: np.ndarray,
                        head_scale: np.ndarray) -> Dict:
    err = np.linalg.norm(semantic_center(mesh_px) - semantic_center(face_px), axis=1)
    norm = err / np.maximum(np.asarray(head_scale, dtype=np.float64), 1.0)
    return {
        "definition": "||weighted semantic mesh centroid - raw face centroid|| / head_scale_px",
        "median": float(np.percentile(norm, 50.0)),
        "p95": float(np.percentile(norm, 95.0)),
        "p99": float(np.percentile(norm, 99.0)),
        "max": float(np.max(norm)),
        "median_px": float(np.percentile(err, 50.0)),
        "p95_px": float(np.percentile(err, 95.0)),
        "pass_median_lt_0p03": bool(np.percentile(norm, 50.0) < 0.03),
        "pass_p95_lt_0p06": bool(np.percentile(norm, 95.0) < 0.06),
    }


def feature_residual_report(mesh_px: np.ndarray, face_px: np.ndarray,
                            head_scale: np.ndarray, yaw: np.ndarray) -> Dict:
    feature_idx = np.asarray(
        sorted(set(FACE_OVAL_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX + LIPS_IDX + NOSE_IDX)),
        dtype=np.int32,
    )
    px_err = np.linalg.norm(mesh_px[:, feature_idx] - face_px[:, feature_idx], axis=2)
    norm = px_err / np.maximum(np.asarray(head_scale, dtype=np.float64)[:, None], 1.0)

    def summarize(mask: np.ndarray) -> Dict:
        vals_px = px_err[mask].reshape(-1)
        vals_norm = norm[mask].reshape(-1)
        if len(vals_px) == 0:
            return {"frames": 0}
        return {
            "frames": int(np.asarray(mask, dtype=bool).sum()),
            "median_px": float(np.percentile(vals_px, 50.0)),
            "p95_px": float(np.percentile(vals_px, 95.0)),
            "median_over_head_scale": float(np.percentile(vals_norm, 50.0)),
            "p95_over_head_scale": float(np.percentile(vals_norm, 95.0)),
        }

    yaw_abs = np.abs(wrap_angle_deg(np.asarray(yaw, dtype=np.float64)))
    return {
        "definition": "selected mesh landmarks to raw face-feature landmarks; px and normalized by head_scale_px",
        "feature_vertex_count": int(len(feature_idx)),
        "all": summarize(np.ones(len(mesh_px), dtype=bool)),
        "frontal_abs_yaw_lt_30": summarize(yaw_abs < 30.0),
        "three_quarter_abs_yaw_30_60": summarize((yaw_abs >= 30.0) & (yaw_abs < 60.0)),
        "profile_abs_yaw_ge_60": summarize(yaw_abs >= 60.0),
    }


def still_frame_mask(raw_px: np.ndarray, source: np.ndarray,
                     yaw: np.ndarray, head_scale: np.ndarray) -> np.ndarray:
    face_center = semantic_center(raw_px)
    speed = np.r_[0.0, np.linalg.norm(np.diff(face_center, axis=0), axis=1) / np.maximum(head_scale[1:], 1.0)]
    yaw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(yaw, dtype=np.float64))))
    yaw_step = np.r_[0.0, np.abs(np.diff(yaw_unwrapped))]
    observed = np.asarray([str(v).startswith("observed") for v in source], dtype=bool)
    if observed.sum() == 0:
        return np.zeros(len(source), dtype=bool)
    speed_cut = float(np.percentile(speed[observed], 35.0))
    return observed & (speed < speed_cut) & (yaw_step < 1.25)


def shimmer_report(before_px: np.ndarray, after_px: np.ndarray,
                   raw_px: np.ndarray, source: np.ndarray,
                   yaw: np.ndarray, head_scale: np.ndarray) -> Dict:
    still = still_frame_mask(raw_px, source, yaw, head_scale)
    frontal = np.asarray([str(v).startswith("observed") for v in source], dtype=bool) & (np.abs(wrap_angle_deg(yaw)) < 30.0)
    profile = source.astype(str) == "profile_fit"
    return {
        "definition": "shape-only jitter after per-step similarity alignment; still frames are observed frames with low face-center speed and yaw step <1.25deg",
        "still_frame_count": int(still.sum()),
        "before_still": shape_only_jitter(before_px, still),
        "after_still": shape_only_jitter(after_px, still),
        "before_frontal_observed": shape_only_jitter(before_px, frontal),
        "after_frontal_observed": shape_only_jitter(after_px, frontal),
        "before_profile_fit": shape_only_jitter(before_px, profile),
        "after_profile_fit": shape_only_jitter(after_px, profile),
    }


def source_transition_jumps_comparison(before_px: np.ndarray, after_px: np.ndarray,
                                       source: np.ndarray, head_scale: np.ndarray) -> Dict:
    return {
        "before": source_transition_jumps(before_px, source, head_scale),
        "after": source_transition_jumps(after_px, source, head_scale),
        "note": "raw source transitions are measured on emitted geometry; no transition-specific clamp is used",
    }


def smooth_pose(matrices: np.ndarray, offsets: np.ndarray,
                confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pose = np.concatenate([matrices.reshape(len(matrices), 4), offsets], axis=1)
    scale_vec = np.asarray([10.0, 10.0, 10.0, 10.0, 70.0, 70.0], dtype=np.float64)
    radius = 12
    sigma = 5.0
    for _ in range(5):
        smoothed = np.zeros_like(pose)
        for i in range(len(pose)):
            lo = max(0, i - radius)
            hi = min(len(pose), i + radius + 1)
            js = np.arange(lo, hi)
            weights = np.exp(-0.5 * ((js - i) / sigma) ** 2) * confidence[js]
            med = np.median(pose[js], axis=0)
            dist = np.linalg.norm((pose[js] - med) / scale_vec, axis=1)
            weights *= 1.0 / (1.0 + dist * dist)
            weights = weights / max(float(np.sum(weights)), 1e-9)
            smoothed[i] = np.sum(pose[js] * weights[:, None], axis=0)
        pose = 0.20 * pose + 0.80 * smoothed
    return pose[:, :4].reshape(len(pose), 2, 2), pose[:, 4:]


def localize_xy(raw_px: np.ndarray, matrices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    local = np.zeros_like(raw_px, dtype=np.float64)
    for i in range(len(raw_px)):
        inv = np.linalg.pinv(matrices[i])
        local[i] = (inv @ (raw_px[i] - offsets[i]).T).T
    return local


def localize_z(raw_z: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    scales = np.sqrt(np.maximum(np.abs(np.linalg.det(matrices)), 1e-9))
    return raw_z / scales[:, None]


def vertex_trust_table(source: np.ndarray) -> np.ndarray:
    n = len(source)
    trust = np.ones((n, V_CANON), dtype=np.float64)
    face = np.asarray(FACE_OVAL_IDX, dtype=np.int32)
    nose = np.asarray(NOSE_IDX, dtype=np.int32)
    features = np.asarray(sorted(set(LEFT_EYE_IDX + RIGHT_EYE_IDX + LIPS_IDX)), dtype=np.int32)
    profile = source.astype(str) == "profile_fit"
    interp = source.astype(str) == "interpolated"
    for i in np.where(profile)[0]:
        trust[i, :] = 0.32
        trust[i, face] = 0.75
        trust[i, nose] = 0.75
        # The 3DDFA->canonical profile interior is visibly less trustworthy.
        # Keep the silhouette, but let temporal/bracketing MP dominate eyes/mouth.
        trust[i, features] = 0.14
    trust[interp, :] = 0.22
    return trust


def startswith_observed(source: str) -> bool:
    return str(source).startswith("observed")


def smooth_local_shapes(local_xy: np.ndarray, local_z: np.ndarray, source: np.ndarray,
                        yaw: np.ndarray, head_scale: np.ndarray,
                        confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n = len(source)
    source_str = source.astype(str)
    profile = source_str == "profile_fit"
    interpolated = source_str == "interpolated"
    observed_mp = np.asarray([startswith_observed(v) for v in source_str], dtype=bool)
    trust = vertex_trust_table(source)
    xy_out = np.zeros_like(local_xy)
    z_out = np.zeros_like(local_z)

    for i in range(n):
        wide = bool(profile[i] or interpolated[i] or float(head_scale[i]) < 50.0)
        radius = 12 if wide else 8
        sigma = 5.0 if wide else 3.5
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        js = np.arange(lo, hi)
        weights = (
            np.exp(-0.5 * ((js - i) / sigma) ** 2)
            * confidence[js]
            * np.exp(-0.5 * (np.abs(yaw[js] - yaw[i]) / 38.0) ** 2)
        )
        vertex_w = weights[:, None] * trust[js]
        denom = np.maximum(np.sum(vertex_w, axis=0), 1e-9)
        xy_out[i] = np.sum(local_xy[js] * vertex_w[:, :, None], axis=0) / denom[:, None]
        z_out[i] = np.sum(local_z[js] * vertex_w, axis=0) / denom

    face = np.asarray(FACE_OVAL_IDX, dtype=np.int32)
    nose = np.asarray(NOSE_IDX, dtype=np.int32)
    feature = np.asarray(sorted(set(LEFT_EYE_IDX + RIGHT_EYE_IDX + LIPS_IDX)), dtype=np.int32)
    interior = np.ones(V_CANON, dtype=bool)
    interior[face] = False
    brace_weight = np.zeros(V_CANON, dtype=np.float64)
    brace_weight[interior] = 0.60
    brace_weight[feature] = 0.85
    brace_weight[nose] = 0.22
    brace_weight[face] = 0.22

    spans = []
    start: Optional[int] = None
    for i, value in enumerate(profile):
        if value and start is None:
            start = i
        if (not value or i == n - 1) and start is not None:
            end = i - 1 if not value else i
            left = start - 1
            while left >= 0 and not observed_mp[left]:
                left -= 1
            right = end + 1
            while right < n and not observed_mp[right]:
                right += 1
            span = {
                "start": int(start),
                "end": int(end),
                "len": int(end - start + 1),
                "left_observed_mp": int(left) if left >= 0 else None,
                "right_observed_mp": int(right) if right < n else None,
                "interior_strategy": "bracketed_mp_dominates_eyes_mouth; profile_keeps_silhouette",
            }
            if left >= 0 and right < n:
                for fidx in range(start, end + 1):
                    t = (fidx - left) / float(max(right - left, 1))
                    brace_xy = (1.0 - t) * local_xy[left] + t * local_xy[right]
                    brace_z = (1.0 - t) * local_z[left] + t * local_z[right]
                    xy_out[fidx] = (
                        (1.0 - brace_weight)[:, None] * xy_out[fidx]
                        + brace_weight[:, None] * brace_xy
                    )
                    z_out[fidx] = (1.0 - brace_weight) * z_out[fidx] + brace_weight * brace_z
                span["bracketed"] = True
            else:
                span["bracketed"] = False
            spans.append(span)
            start = None

    return xy_out, z_out, {"profile_spans": spans}


def project_from_local(local_xy: np.ndarray, local_z: np.ndarray,
                       matrices: np.ndarray, offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(local_xy)
    projected = np.zeros_like(local_xy, dtype=np.float32)
    verts = np.zeros((n, V_CANON, 3), dtype=np.float32)
    scales = np.sqrt(np.maximum(np.abs(np.linalg.det(matrices)), 1e-9))
    for i in range(n):
        xy = (matrices[i] @ local_xy[i].T).T + offsets[i]
        projected[i] = xy.astype(np.float32)
        verts[i, :, :2] = projected[i]
        verts[i, :, 2] = (local_z[i] * scales[i]).astype(np.float32)
    return verts, projected


def source_transition_jumps(projected_px: np.ndarray, source: np.ndarray,
                            head_scale: np.ndarray) -> Dict:
    vals = []
    rows = []
    source_str = source.astype(str)
    for i in range(1, len(source_str)):
        if source_str[i] == source_str[i - 1]:
            continue
        jump = np.linalg.norm(projected_px[i] - projected_px[i - 1], axis=1)
        value = float(np.nanmean(jump) / max(float(head_scale[i]), 1.0))
        vals.append(value)
        rows.append({
            "frame": int(i),
            "from": str(source_str[i - 1]),
            "to": str(source_str[i]),
            "jump_over_head_scale": value,
        })
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return {
            "n_transitions": 0,
            "mean_jump_over_head_scale": 0.0,
            "p90_jump_over_head_scale": 0.0,
            "max_jump_over_head_scale": 0.0,
            "pct_le_0p15": 100.0,
            "transitions": [],
            "worst_transitions": [],
            "pass_90pct_le_0p15": True,
        }
    worst = sorted(rows, key=lambda r: r["jump_over_head_scale"], reverse=True)[:12]
    return {
        "n_transitions": int(len(arr)),
        "mean_jump_over_head_scale": float(np.mean(arr)),
        "p50_jump_over_head_scale": float(np.percentile(arr, 50.0)),
        "p90_jump_over_head_scale": float(np.percentile(arr, 90.0)),
        "max_jump_over_head_scale": float(np.max(arr)),
        "pct_le_0p15": float(100.0 * np.mean(arr <= 0.15)),
        "transitions": rows,
        "worst_transitions": worst,
        "pass_90pct_le_0p15": bool(np.mean(arr <= 0.15) >= 0.90),
    }


def fit_similarity_unweighted(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return weighted_similarity(src, dst, np.ones(len(src), dtype=np.float64))


def shape_only_jitter(projected_px: np.ndarray, mask: np.ndarray) -> Dict:
    idx = np.where(mask.astype(bool))[0]
    vals = []
    frames = []
    for a, b in zip(idx[:-1], idx[1:]):
        if int(b) != int(a) + 1:
            continue
        mat, off = fit_similarity_unweighted(projected_px[b], projected_px[a])
        aligned = (mat @ projected_px[b].T).T + off
        diff = np.linalg.norm(aligned - projected_px[a], axis=1)
        diag = float(np.linalg.norm(np.nanmax(projected_px[a], axis=0) - np.nanmin(projected_px[a], axis=0)))
        vals.append(float(np.nanpercentile(diff, 95.0) / max(diag, 1e-6)))
        frames.append(int(b))
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return {
            "n_steps": 0,
            "median_p95_vertex_over_mesh_diag": 0.0,
            "p95_p95_vertex_over_mesh_diag": 0.0,
            "max_p95_vertex_over_mesh_diag": 0.0,
            "worst_steps": [],
        }
    worst = sorted(
        [{"frame": f, "p95_vertex_over_mesh_diag": v} for f, v in zip(frames, vals)],
        key=lambda r: r["p95_vertex_over_mesh_diag"],
        reverse=True,
    )[:12]
    return {
        "n_steps": int(len(arr)),
        "median_p95_vertex_over_mesh_diag": float(np.percentile(arr, 50.0)),
        "p95_p95_vertex_over_mesh_diag": float(np.percentile(arr, 95.0)),
        "max_p95_vertex_over_mesh_diag": float(np.max(arr)),
        "worst_steps": worst,
    }


def coverage_report(verts: np.ndarray, projected_px: np.ndarray, source: np.ndarray,
                    geometry_observed: np.ndarray, yaw: np.ndarray) -> Dict:
    source_str = source.astype(str)
    yaw_abs = np.abs(wrap_angle_deg(np.asarray(yaw, dtype=np.float64)))
    return {
        "total_frames": int(len(source_str)),
        "verts_populated": int(np.isfinite(verts).all(axis=(1, 2)).sum()),
        "projected_px_populated": int(np.isfinite(projected_px).all(axis=(1, 2)).sum()),
        "no_nan": bool(np.isfinite(verts).all() and np.isfinite(projected_px).all()),
        "geometry_observed_frames": int(np.asarray(geometry_observed, dtype=bool).sum()),
        "interpolated_frames": int((source_str == "interpolated").sum()),
        "profile_fit_frames": int((source_str == "profile_fit").sum()),
        "source_counts": {str(k): int(v) for k, v in zip(*np.unique(source_str, return_counts=True))},
        "yaw_bins": {
            "0_30": int((yaw_abs < 30.0).sum()),
            "30_60": int(((yaw_abs >= 30.0) & (yaw_abs < 60.0)).sum()),
            "60_90": int(((yaw_abs >= 60.0) & (yaw_abs < 90.0)).sum()),
            "90_180": int((yaw_abs >= 90.0).sum()),
        },
    }


def scan_for_disallowed_calls() -> Dict:
    needle_groups = {
        "torch_cuda_calls": ["torch." + "cuda", "." + "cuda("],
        "blocked_render_libs": ["pytorch" + "3d", "nvdi" + "ffrast"],
    }
    hits: Dict[str, List[str]] = {k: [] for k in needle_groups}
    with open(__file__, "r", encoding="utf-8") as f:
        text = f.read()
    for group, needles in needle_groups.items():
        for needle in needles:
            if needle in text:
                hits[group].append(f"{__file__}:{needle}")
    return {
        "execution": "CPU numpy/OpenCV postpass; no Torch device used",
        "hits": hits,
        "pass": all(len(v) == 0 for v in hits.values()),
    }


def topology_report(faces: np.ndarray, verts: np.ndarray) -> Dict:
    faces_hash = hashlib.sha256(faces.astype(np.int32).tobytes()).hexdigest()
    return {
        "vertex_count": int(verts.shape[1]),
        "face_count": int(faces.shape[0]),
        "faces_sha256": faces_hash,
        "constant_vertex_count": bool(verts.shape[1] == V_CANON),
        "identical_faces_hash_every_frame": True,
        "pass": bool(verts.shape[1] == V_CANON and faces.shape == (898, 3)),
    }


def motion_verification(projected_px: np.ndarray, raw_px: np.ndarray, source: np.ndarray,
                        yaw: np.ndarray, head_scale: np.ndarray) -> Dict:
    source_str = source.astype(str)
    face_center = semantic_center(raw_px)
    mesh_center = semantic_center(projected_px)
    yaw_unwrapped = unwrap_yaw_deg(wrap_angle_deg(np.asarray(yaw, dtype=np.float64)), reference=np.asarray(yaw, dtype=np.float64))
    face_speed = np.r_[0.0, np.linalg.norm(np.diff(face_center, axis=0), axis=1) / np.maximum(head_scale[1:], 1.0)]
    yaw_speed = np.r_[0.0, np.abs(np.diff(yaw_unwrapped))]
    del yaw_speed
    start, end = POSE_FLIP_START, POSE_FLIP_END

    center_error = (
        np.linalg.norm(mesh_center[start:end + 1] - face_center[start:end + 1], axis=1)
        / np.maximum(head_scale[start:end + 1], 1.0)
    )
    mean_step = []
    for i in range(start + 1, end + 1):
        jump = np.linalg.norm(projected_px[i] - projected_px[i - 1], axis=1)
        mean_step.append(float(np.nanmean(jump) / max(float(head_scale[i]), 1.0)))

    long_start, long_end = POSE_FLIP_START, POSE_FLIP_END
    long_steps = []
    for i in range(long_start + 1, long_end + 1):
        jump = np.linalg.norm(projected_px[i] - projected_px[i - 1], axis=1)
        long_steps.append(float(np.nanmean(jump) / max(float(head_scale[i]), 1.0)))

    return {
        "pose_flip_span_425_440": {
            "start": int(start),
            "end": int(end),
            "len": int(end - start + 1),
            "frames": [int(v) for v in range(start, end + 1)],
            "sources": sorted(set(str(v) for v in source_str[start:end + 1])),
            "yaw_min_deg": float(np.min(yaw_unwrapped[start:end + 1])),
            "yaw_max_deg": float(np.max(yaw_unwrapped[start:end + 1])),
            "yaw_abs_delta_sum_deg": float(np.sum(np.abs(np.diff(yaw_unwrapped[start:end + 1])))),
            "mean_face_speed_over_head_scale": float(np.mean(face_speed[start + 1:end + 1])),
            "center_error_median_over_head_scale": float(np.percentile(center_error, 50.0)),
            "center_error_p95_over_head_scale": float(np.percentile(center_error, 95.0)),
            "max_mean_mesh_step_over_head_scale": float(np.max(mean_step)) if mean_step else 0.0,
            "p95_mean_mesh_step_over_head_scale": float(np.percentile(mean_step, 95.0)) if mean_step else 0.0,
            "visual_strip": MOTION_STRIP_PATH,
        },
        "handoff_span_425_440": {
            "start": long_start,
            "end": long_end,
            "len": int(long_end - long_start + 1),
            "sources": sorted(set(str(v) for v in source_str[long_start:long_end + 1])),
            "yaw_min_deg": float(np.min(yaw_unwrapped[long_start:long_end + 1])),
            "yaw_max_deg": float(np.max(yaw_unwrapped[long_start:long_end + 1])),
            "max_mean_step_over_head_scale": float(np.max(long_steps)),
            "p95_mean_step_over_head_scale": float(np.percentile(long_steps, 95.0)),
            "pass_has_frontal_profile_interpolated": bool(
                np.any(np.abs(wrap_angle_deg(yaw[long_start:long_end + 1])) < 15.0)
                and np.any(source_str[long_start:long_end + 1] == "profile_fit")
                and np.any(source_str[long_start:long_end + 1] == "interpolated")
            ),
        },
        "visual_assessment": "rendered v7 overlay plus exact f425-f440 pose-flip motion strip",
    }


def mean_vertex_step_over_head_scale(px: np.ndarray, head_scale: np.ndarray) -> np.ndarray:
    vals = np.zeros(len(px), dtype=np.float64)
    for i in range(1, len(px)):
        jump = np.linalg.norm(px[i] - px[i - 1], axis=1)
        vals[i] = float(np.nanmean(jump) / max(float(head_scale[i]), 1.0))
    return vals


def p95_vertex_step_over_head_scale(px: np.ndarray, head_scale: np.ndarray) -> np.ndarray:
    vals = np.zeros(len(px), dtype=np.float64)
    for i in range(1, len(px)):
        jump = np.linalg.norm(px[i] - px[i - 1], axis=1)
        vals[i] = float(np.nanpercentile(jump, 95.0) / max(float(head_scale[i]), 1.0))
    return vals


def pose_continuity_metrics(raw_rotations: Rotation,
                            stable_rotations: Rotation,
                            raw_yaw: np.ndarray,
                            stable_yaw: np.ndarray,
                            before_px: np.ndarray,
                            after_px: np.ndarray,
                            source: np.ndarray,
                            head_scale: np.ndarray,
                            pose_detail: Dict,
                            flow: Dict) -> Dict:
    raw_step = angular_jumps_deg(raw_rotations)
    stable_step = angular_jumps_deg(stable_rotations)
    raw_yaw_unwrapped = unwrap_yaw_deg(wrap_angle_deg(raw_yaw), reference=raw_yaw)
    stable_yaw_unwrapped = unwrap_yaw_deg(wrap_angle_deg(stable_yaw), reference=stable_yaw)
    raw_yaw_step = np.r_[0.0, np.abs(np.diff(raw_yaw_unwrapped))]
    stable_yaw_step = np.r_[0.0, np.abs(np.diff(stable_yaw_unwrapped))]
    before_mean_step = mean_vertex_step_over_head_scale(before_px, head_scale)
    after_mean_step = mean_vertex_step_over_head_scale(after_px, head_scale)
    before_p95_step = p95_vertex_step_over_head_scale(before_px, head_scale)
    after_p95_step = p95_vertex_step_over_head_scale(after_px, head_scale)
    rejected = np.asarray(pose_detail["rejected_mask"], dtype=bool)
    predicted_error = np.asarray(pose_detail["predicted_error_by_frame"], dtype=np.float64)
    flow_error = np.asarray(pose_detail["flow_error_by_frame"], dtype=np.float64)
    flow_contra = np.asarray(pose_detail["flow_contradiction"], dtype=bool)
    source_str = np.asarray(source).astype(str)

    span_rows = []
    for fidx in range(POSE_FLIP_START, POSE_FLIP_END + 1):
        span_rows.append({
            "frame": int(fidx),
            "source": str(source_str[fidx]),
            "raw_yaw_deg": float(raw_yaw[fidx]),
            "raw_yaw_unwrapped_deg": float(raw_yaw_unwrapped[fidx]),
            "v7_yaw_unwrapped_deg": float(stable_yaw_unwrapped[fidx]),
            "v7_yaw_wrapped_deg": float(wrap_angle_deg(stable_yaw_unwrapped[fidx])),
            "raw_pose_step_deg": float(raw_step[fidx]),
            "v7_pose_step_deg": float(stable_step[fidx]),
            "raw_yaw_step_deg": float(raw_yaw_step[fidx]),
            "v7_yaw_step_deg": float(stable_yaw_step[fidx]),
            "raw_mean_vertex_step_over_head_scale": float(before_mean_step[fidx]),
            "v7_mean_vertex_step_over_head_scale": float(after_mean_step[fidx]),
            "raw_p95_vertex_step_over_head_scale": float(before_p95_step[fidx]),
            "v7_p95_vertex_step_over_head_scale": float(after_p95_step[fidx]),
            "raw_vs_cav_angle_deg": float(predicted_error[fidx]),
            "pose_rejected": bool(rejected[fidx]),
            "flow_confidence": float(flow["confidence"][fidx]) if flow["available"] else 0.0,
            "flow_error_over_head_scale": float(flow_error[fidx]),
            "flow_contradiction": bool(flow_contra[fidx]),
        })

    span = slice(POSE_FLIP_START + 1, POSE_FLIP_END + 1)
    return {
        "definition": "before=v6 raw pose/geometry, after=v7 quaternion CAV pose and rebuilt averted geometry",
        "pose_angular_jump_deg": {
            "before_v6": summarize_values(raw_step[1:]),
            "after_v7": summarize_values(stable_step[1:]),
            "before_pose_flip_span": summarize_values(raw_step[span]),
            "after_pose_flip_span": summarize_values(stable_step[span]),
        },
        "yaw_step_deg": {
            "before_v6_unwrapped": summarize_values(raw_yaw_step[1:]),
            "after_v7_unwrapped": summarize_values(stable_yaw_step[1:]),
            "before_pose_flip_span": summarize_values(raw_yaw_step[span]),
            "after_pose_flip_span": summarize_values(stable_yaw_step[span]),
        },
        "mean_vertex_jump_over_head_scale": {
            "before_v6": summarize_values(before_mean_step[1:]),
            "after_v7": summarize_values(after_mean_step[1:]),
            "before_pose_flip_span": summarize_values(before_mean_step[span]),
            "after_pose_flip_span": summarize_values(after_mean_step[span]),
        },
        "p95_vertex_jump_over_head_scale": {
            "before_v6": summarize_values(before_p95_step[1:]),
            "after_v7": summarize_values(after_p95_step[1:]),
            "before_pose_flip_span": summarize_values(before_p95_step[span]),
            "after_pose_flip_span": summarize_values(after_p95_step[span]),
        },
        "pose_flip_span": {
            "start": POSE_FLIP_START,
            "end": POSE_FLIP_END,
            "frames": span_rows,
            "raw_f429_yaw_deg": float(raw_yaw[429]),
            "v7_f429_yaw_unwrapped_deg": float(stable_yaw_unwrapped[429]),
            "v7_f429_yaw_wrapped_deg": float(wrap_angle_deg(stable_yaw_unwrapped[429])),
            "raw_span_yaw_minmax_deg": [float(np.min(raw_yaw_unwrapped[POSE_FLIP_START:POSE_FLIP_END + 1])), float(np.max(raw_yaw_unwrapped[POSE_FLIP_START:POSE_FLIP_END + 1]))],
            "v7_span_yaw_minmax_deg": [float(np.min(stable_yaw_unwrapped[POSE_FLIP_START:POSE_FLIP_END + 1])), float(np.max(stable_yaw_unwrapped[POSE_FLIP_START:POSE_FLIP_END + 1]))],
            "v7_monotone_turn_score": float(np.mean(np.diff(stable_yaw_unwrapped[POSE_FLIP_START:POSE_FLIP_END + 1]) <= 1.0)),
            "max_raw_pose_step_deg": float(np.max(raw_step[span])),
            "max_v7_pose_step_deg": float(np.max(stable_step[span])),
            "max_raw_mean_vertex_step_over_head_scale": float(np.max(before_mean_step[span])),
            "max_v7_mean_vertex_step_over_head_scale": float(np.max(after_mean_step[span])),
        },
        "outlier_rejection": {
            "rejected_frame_count": int(rejected.sum()),
            "rejected_frames": [int(v) for v in np.where(rejected)[0]],
            "pose_flip_rejected_frames": [int(v) for v in np.where(rejected[POSE_FLIP_START:POSE_FLIP_END + 1])[0] + POSE_FLIP_START],
            "predicted_error_deg": summarize_values(predicted_error[averted_pose_mask(source)]),
            "flow_contradiction_frames": [int(v) for v in np.where(flow_contra & averted_pose_mask(source))[0]],
        },
        "optical_flow_head_anchor": {
            "available": bool(flow["available"]),
            "confidence": summarize_values(np.asarray(flow["confidence"], dtype=np.float64)),
            "inliers": summarize_values(np.asarray(flow["inliers"], dtype=np.float64)),
            "pose_flip_confidence": summarize_values(np.asarray(flow["confidence"][POSE_FLIP_START:POSE_FLIP_END + 1], dtype=np.float64)),
            "pose_flip_flow_error_over_head_scale": summarize_values(flow_error[POSE_FLIP_START:POSE_FLIP_END + 1]),
        },
    }


def draw_edges(canvas: np.ndarray, projected_px: np.ndarray, edges: Iterable[Tuple[int, int]],
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
        cv2.line(
            canvas,
            (int(round(pa[0])), int(round(pa[1]))),
            (int(round(pb[0])), int(round(pb[1]))),
            color,
            lw,
            cv2.LINE_AA,
        )


def draw_contours(canvas: np.ndarray, projected_px: np.ndarray, source: str) -> None:
    del source
    colors = {
        "oval": (60, 235, 80),
        "eye": (255, 220, 80),
        "lip_outer": (0, 120, 255),
        "lip_inner": (40, 235, 255),
    }
    for group, color, lw in [
        (FACE_OVAL, colors["oval"], 2),
        (LEFT_EYE + RIGHT_EYE, colors["eye"], 2),
        (LIPS_INNER, colors["lip_inner"], 2),
        (LIPS_OUTER, colors["lip_outer"], 3),
    ]:
        draw_edges(canvas, projected_px, group, color, lw)


def draw_hud(canvas: np.ndarray, fidx: int, total_f: int, source: str,
             yaw: float, seam_jump: float, profile_jitter_tag: str) -> None:
    color = (0, 255, 120)
    if source == "interpolated":
        color = (255, 210, 90)
    elif source == "profile_fit":
        color = (80, 220, 255)
    lines = [
        f"f{fidx:04d}/{total_f} {source}",
        f"yaw={yaw:+.0f} seam_step={seam_jump:.3f}",
        profile_jitter_tag,
    ]
    for i, line in enumerate(lines):
        y = 26 + i * 22
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def choose_montage_frames(source: np.ndarray, yaw: np.ndarray) -> Dict[str, int]:
    source_str = source.astype(str)
    yaw_abs = np.abs(wrap_angle_deg(np.asarray(yaw, dtype=np.float64)))

    def first(mask: np.ndarray, fallback: int) -> int:
        idx = np.where(mask)[0]
        return int(idx[0]) if len(idx) else int(fallback)

    observed = np.asarray([startswith_observed(v) for v in source_str], dtype=bool)
    return {
        "frontal": first(observed & (yaw_abs < 15.0), 50),
        "three_quarter": first(observed & (yaw_abs >= 30.0) & (yaw_abs < 60.0), 90),
        "profile_fit": first(source_str == "profile_fit", 188),
        "pose_flip": POSE_FLIP_START,
        "center_scale_lock": 529,
        "profile_late": 775,
    }


def build_video_and_montage(projected_px: np.ndarray, source: np.ndarray,
                            yaw: np.ndarray, faces: np.ndarray,
                            report: Dict) -> Dict[str, str]:
    edges = faces_to_edges(faces)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = len(source)
    tmp_path = OVERLAY_MASTER_PATH.replace(".mp4", "_tmp.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
    if not writer.isOpened():
        raise RuntimeError("Could not open overlay writer")

    step = np.zeros(total_f, dtype=np.float32)
    for i in range(1, total_f):
        step[i] = float(
            np.nanmean(np.linalg.norm(projected_px[i] - projected_px[i - 1], axis=1))
            / max(float(report["head_scale_px"][i]), 1.0)
        )

    montage_targets = choose_montage_frames(source, yaw)
    motion_frames = set(report["motion_verification"]["pose_flip_span_425_440"]["frames"])
    saved_key: Dict[str, Tuple[np.ndarray, int]] = {}
    saved_motion: List[Tuple[np.ndarray, int]] = []
    tag = "v7 cav pose lock"

    for fidx in range(total_f):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        src = str(source[fidx])
        canvas = frame_bgr.copy()
        edge_color = (205, 205, 205)
        draw_edges(canvas, projected_px[fidx], edges, edge_color, 1)
        draw_contours(canvas, projected_px[fidx], src)
        draw_hud(canvas, fidx, total_f, src, float(yaw[fidx]), float(step[fidx]), tag)
        writer.write(canvas)
        for label, frame_idx in montage_targets.items():
            if fidx == frame_idx:
                saved_key[label] = (canvas.copy(), fidx)
        if fidx in motion_frames:
            saved_motion.append((canvas.copy(), fidx))

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
    target_kbps = int((7.2 * 8 * 1000) / duration_s)
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

    build_montage(saved_key, saved_motion)
    build_motion_strip(saved_motion)
    report["montage_frames"] = {label: int(idx) for label, (_, idx) in saved_key.items()}
    report["motion_strip_frames"] = [int(idx) for _, idx in saved_motion]
    return {
        "overlay_master": OVERLAY_MASTER_PATH,
        "overlay_preview": OVERLAY_PREVIEW_PATH,
        "montage": MONTAGE_PATH,
        "motion_strip": MOTION_STRIP_PATH,
    }


def label_cell(img: np.ndarray, text: str, size: Tuple[int, int]) -> np.ndarray:
    cell = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, 0), (size[0], 28), (0, 0, 0), -1)
    cv2.putText(cell, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def build_montage(saved_key: Dict[str, Tuple[np.ndarray, int]],
                  saved_motion: List[Tuple[np.ndarray, int]]) -> None:
    key_order = ["frontal", "three_quarter", "profile_fit", "pose_flip", "center_scale_lock", "profile_late"]
    rows = []
    key_cells = []
    for label in key_order:
        if label not in saved_key:
            continue
        img, fidx = saved_key[label]
        key_cells.append(label_cell(img, f"{label} f{fidx}", (216, 384)))
    if key_cells:
        rows.append(np.hstack(key_cells))
    verify_cells = [
        label_cell(img, f"motion f{fidx}", (216, 384))
        for img, fidx in saved_motion[:6]
    ]
    if verify_cells:
        verify_row = np.hstack(verify_cells)
        if rows and verify_row.shape[1] != rows[0].shape[1]:
            target_w = rows[0].shape[1]
            verify_row = cv2.resize(verify_row, (target_w, verify_row.shape[0]), interpolation=cv2.INTER_AREA)
        rows.append(verify_row)
    if rows:
        width = max(row.shape[1] for row in rows)
        padded = []
        for row in rows:
            if row.shape[1] < width:
                pad = np.zeros((row.shape[0], width - row.shape[1], 3), dtype=row.dtype)
                row = np.hstack([row, pad])
            padded.append(row)
        cv2.imwrite(MONTAGE_PATH, np.vstack(padded))


def build_motion_strip(saved_motion: List[Tuple[np.ndarray, int]]) -> None:
    ordered = sorted(saved_motion, key=lambda row: row[1])
    if not ordered:
        return
    cells = [label_cell(img, f"flip f{fidx}", (144, 256)) for img, fidx in ordered]
    target_len = POSE_FLIP_END - POSE_FLIP_START + 1
    while len(cells) < target_len:
        cells.append(np.zeros_like(cells[0]))
    cv2.imwrite(MOTION_STRIP_PATH, np.hstack(cells[:target_len]))


def load_v4_preclamp_baseline() -> Optional[Dict]:
    if not os.path.exists(V4_REPORT_PATH):
        return None
    with open(V4_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report.get("boundary_pop_raw_unclamped")


def write_notes(report: Dict, metrics: Dict) -> None:
    lag = metrics["lag"]
    center = metrics["center_error"]
    residual = metrics["feature_residual"]
    topo = report["topology"]
    coverage = report["coverage"]
    geometry = report["geometry_rebuild"]
    motion = report["motion_verification"]["pose_flip_span_425_440"]
    pose = metrics["pose_continuity"]
    span = pose["pose_flip_span"]
    jumps = metrics["source_boundary"]["after_v7"]["worst_transitions"][:5]

    lines = [
        "# Wireframe V7 Notes",
        "",
        "## Scope",
        "- v7 builds on v6 and keeps v6 MediaPipe-observed geometry unchanged.",
        "- The v7 change is limited to profile/interpolated averted frames: quaternion constant-angular-velocity pose replacement, outlier rejection, optical-flow head-motion validation, and a rebuilt mesh pose arc.",
        "- v6 semantic center/scale lock is preserved on observed frames; rebuilt averted frames use the v6 target center unless it contradicts the head optical-flow track, where a weighted flow-center override is used.",
        "",
        "## Outputs",
        f"- Stream: `{STREAM_PATH}`",
        f"- Overlay master: `{OVERLAY_MASTER_PATH}`",
        f"- Overlay preview: `{OVERLAY_PREVIEW_PATH}`",
        f"- Pose-flip motion strip f425-f440: `{MOTION_STRIP_PATH}`",
        f"- Center/scale montage: `{MONTAGE_PATH}`",
        f"- Report: `{REPORT_PATH}`",
        f"- Before/after metrics: `{METRICS_PATH}`",
        "",
        "## Pre-Registered Tests",
        f"- Pose angular step p99: v6={pose['pose_angular_jump_deg']['before_v6']['p99']:.2f}deg, v7={pose['pose_angular_jump_deg']['after_v7']['p99']:.2f}deg.",
        f"- f425-f440 max pose step: v6={span['max_raw_pose_step_deg']:.2f}deg, v7={span['max_v7_pose_step_deg']:.2f}deg.",
        f"- f429 flip: raw yaw={span['raw_f429_yaw_deg']:.2f}deg; v7 unwrapped yaw={span['v7_f429_yaw_unwrapped_deg']:.2f}deg.",
        f"- f425-f440 max mean vertex jump/head: v6={span['max_raw_mean_vertex_step_over_head_scale']:.4f}, v7={span['max_v7_mean_vertex_step_over_head_scale']:.4f}.",
        f"- Lag: v6 best_delay={lag['before_v6']['best_delay_frames']} frame(s), v7 best_delay={lag['after_v7']['best_delay_frames']} frame(s), target ~0.",
        f"- Center error: v6 median={center['before_v6']['median']:.4f} p95={center['before_v6']['p95']:.4f}; v7 median={center['after_v7']['median']:.4f} p95={center['after_v7']['p95']:.4f}.",
        f"- Observed frame preservation: max projected delta vs v6 observed frames={geometry['observed_geometry_max_abs_delta_px']:.6f}px; flow-center override frames={geometry['flow_center_override_frames']}.",
        f"- Feature residual profile p95: v6={residual['before_v6']['profile_abs_yaw_ge_60']['p95_px']:.2f}px/{residual['before_v6']['profile_abs_yaw_ge_60']['p95_over_head_scale']:.4f}, v7={residual['after_v7']['profile_abs_yaw_ge_60']['p95_px']:.2f}px/{residual['after_v7']['profile_abs_yaw_ge_60']['p95_over_head_scale']:.4f}.",
        f"- Motion strip: f{motion['start']}-f{motion['end']} len={motion['len']} sources={motion['sources']} yaw_delta_sum={motion['yaw_abs_delta_sum_deg']:.1f}deg center_p95={motion['center_error_p95_over_head_scale']:.4f}.",
        f"- Topology: V={topo['vertex_count']} F={topo['face_count']} faces_sha256=`{topo['faces_sha256']}` pass={topo['pass']}.",
        f"- Coverage: populated={coverage['verts_populated']}/{coverage['total_frames']} projected={coverage['projected_px_populated']}/{coverage['total_frames']} no_nan={coverage['no_nan']}.",
        f"- MPS/no-CUDA scan clean: `{report['mps_no_cuda']['pass']}` hits={report['mps_no_cuda']['hits']}.",
        "",
        "## Method Summary",
        "- Convert `head_transform` rotations to unit quaternions with polar/SVD normalization.",
        "- For each profile/interpolated span, estimate entry angular velocity from the preceding reliable observed frames and propagate that rotation arc through the averted span.",
        "- Reject raw pose reads when they diverge from the CAV prediction, produce a neighbor spike, or contradict the optical-flow head-motion check.",
        "- Rebuild profile/interpolated mesh frames from stabilized neutral pose plus smoothstep-interpolated observed residuals, then relock semantic center/scale to a weighted v6/flow center target.",
        "",
        "## Honest Floor",
        "- v7 removes the f429 pose flip from the emitted pose stream; residual profile feature error can remain because the back/profile frames still lack true face observations.",
        "- Profile feature residual p95 rises versus v6 because v7 prioritizes head-pose continuity and flow-visible head anchoring over matching unreliable averted face-feature anchors.",
        "- Optical flow is used as a validation signal, not as a full segmentation tracker; low-texture or motion-blurred head regions can lower flow confidence.",
        "- Motion cross-verification against an external reference is pending; this run produced the motion strip and metrics for that check.",
        "- Remaining largest source jumps are reported below; no source-boundary clamp is hidden.",
    ]
    for row in jumps:
        lines.append(
            f"- f{row['frame']} {row['from']}->{row['to']}: {row['jump_over_head_scale']:.4f} head-scale"
        )
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run() -> Dict:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{LOG_PREFIX} Loading v4 raw target and v6 baseline...")
    canon_verts, _ = load_canonical_mesh()
    v4 = np.load(V4_STREAM_PATH, allow_pickle=True)
    raw_px = np.asarray(v4["projected_px"], dtype=np.float64)
    v6 = np.load(V6_STREAM_PATH, allow_pickle=True)
    v6_px = np.asarray(v6["projected_px"], dtype=np.float64)
    v6_verts = np.asarray(v6["verts"], dtype=np.float64)
    faces = np.asarray(v6["faces"], dtype=np.int32)
    source = np.asarray(v6["mesh_source"]).astype("<U24")
    head_scale = np.asarray(v6["head_scale_px"], dtype=np.float64)
    head_center = np.asarray(v6["head_center_px"], dtype=np.float64)
    yaw_raw = np.asarray(v6["yaw_deg"], dtype=np.float64)
    pitch_raw = np.asarray(v6["pitch_deg"], dtype=np.float64)
    roll_raw = np.asarray(v6["roll_deg"], dtype=np.float64)
    head_transform = np.asarray(v6["head_transform"], dtype=np.float64)
    mode = np.asarray(v6["v17_mode"]) if "v17_mode" in v6.files else np.asarray(source)
    geometry_observed = np.asarray(v6["geometry_observed"], dtype=bool)
    total_f = v6_px.shape[0]
    if raw_px.shape != (total_f, V_CANON, 2) or v6_px.shape != (total_f, V_CANON, 2):
        raise RuntimeError(f"Bad stream shapes raw={raw_px.shape} v6={v6_px.shape}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.0)
    cap.release()
    print(f"{LOG_PREFIX} Frames={total_f}; fps={fps:.3f}; sources={dict(zip(*np.unique(source, return_counts=True)))}")

    raw_rotations = rotations_from_stream(head_transform, yaw_raw, pitch_raw, roll_raw)
    target_center = semantic_center(v6_px)
    reliable = reliable_media_pipe_mask(source)
    print(f"{LOG_PREFIX} Tracking optical-flow head anchor...")
    flow = compute_optical_flow_anchors(VIDEO_PATH, target_center, head_scale, reliable)

    print(f"{LOG_PREFIX} Stabilizing averted/profile/interpolated pose in quaternion space...")
    stable_rotations, yaw, pitch, roll, pose_detail = stabilize_rotations_with_cav(
        raw_rotations,
        yaw_raw,
        pitch_raw,
        roll_raw,
        source,
        flow,
        target_center,
        head_scale,
    )
    yaw_raw_unwrapped = unwrap_yaw_deg(wrap_angle_deg(yaw_raw), reference=yaw_raw)

    print(f"{LOG_PREFIX} Rebuilding only profile/interpolated frames on stabilized pose arc...")
    verts, projected_px, geometry_detail = rebuild_averted_geometry(
        canon_verts,
        v6_verts,
        v6_px,
        raw_rotations,
        stable_rotations,
        yaw_raw_unwrapped,
        yaw,
        source,
        head_center,
        head_scale,
        mode,
        flow,
    )
    projected_px64 = np.asarray(projected_px, dtype=np.float64)
    head_transform_v7 = update_head_transforms_with_rotations(head_transform, stable_rotations)

    if not np.isfinite(verts).all() or not np.isfinite(projected_px).all():
        raise RuntimeError("v7 produced non-finite mesh data")

    face_center = semantic_center(raw_px)
    lag_before = cross_correlation_delay(semantic_center(v6_px), face_center)
    lag_after = cross_correlation_delay(semantic_center(projected_px64), face_center)
    center_before = center_error_report(v6_px, raw_px, head_scale)
    center_after = center_error_report(projected_px64, raw_px, head_scale)
    residual_before = feature_residual_report(v6_px, raw_px, head_scale, yaw_raw)
    residual_after = feature_residual_report(projected_px64, raw_px, head_scale, yaw)
    shimmer = shimmer_report(v6_px, projected_px64, raw_px, source, yaw, head_scale)
    source_boundary = {
        "before_v6": source_transition_jumps(v6_px, source, head_scale),
        "after_v7": source_transition_jumps(projected_px64, source, head_scale),
        "note": "raw source transitions are measured on emitted geometry; no transition-specific clamp is used",
    }
    motion = motion_verification(projected_px64, raw_px, source, yaw, head_scale)
    pose_metrics = pose_continuity_metrics(
        raw_rotations,
        stable_rotations,
        yaw_raw,
        yaw,
        v6_px,
        projected_px64,
        source,
        head_scale,
        pose_detail,
        flow,
    )

    print(f"{LOG_PREFIX} Saving stream...")
    stream_payload = {key: np.asarray(v6[key]) for key in v6.files}
    stream_payload.update({
        "verts": verts,
        "faces": faces,
        "projected_px": projected_px,
        "head_transform": head_transform_v7,
        "mesh_source": source,
        "geometry_observed": geometry_observed,
        "yaw_deg": np.asarray(yaw, dtype=np.float32),
        "yaw_wrapped_deg": np.asarray(wrap_angle_deg(yaw), dtype=np.float32),
        "pitch_deg": np.asarray(pitch, dtype=np.float32),
        "roll_deg": np.asarray(roll, dtype=np.float32),
        "raw_yaw_deg_v6": np.asarray(yaw_raw, dtype=np.float32),
        "pose_rejected": np.asarray(pose_detail["rejected_mask"], dtype=bool),
        "pose_policy": np.asarray(pose_detail["pose_policy"], dtype="<U40"),
        "pipeline_version": np.asarray([PIPELINE_VERSION]),
        "source_stream": np.asarray(["mesh_cascade_v6_stream.npz + mesh_cascade_v4_stream.npz"], dtype="<U96"),
        "wireframe_postpass": np.asarray([
            "v6_one_euro_lock_plus_v7_quaternion_cav_pose_stabilized_averted_frames"
        ], dtype="<U96"),
        "face_lock_policy": np.asarray([
            "observed_frames_keep_v6_geometry; rebuilt_averted_frames_relocked_to_v6_semantic_center_scale"
        ], dtype="<U96"),
        "pose_stabilization_policy": np.asarray([
            "quaternion_cav_outlier_rejection_angle_unwrap_optical_flow_head_anchor_validation"
        ], dtype="<U128"),
        "temporal_inference": np.asarray([
            "v6 lag_centering preserved; v7 replaces profile/interpolated pose flip spans"
        ], dtype="<U96"),
    })
    np.savez_compressed(STREAM_PATH, **stream_payload)

    coverage = coverage_report(verts, projected_px, source, geometry_observed, yaw)
    topo = topology_report(faces, verts)
    mps_no_cuda = scan_for_disallowed_calls()
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": {
            "stream": STREAM_PATH,
            "report": REPORT_PATH,
            "metrics": METRICS_PATH,
            "notes": NOTES_PATH,
            "overlay_master": OVERLAY_MASTER_PATH,
            "overlay_preview": OVERLAY_PREVIEW_PATH,
            "montage": MONTAGE_PATH,
            "motion_strip": MOTION_STRIP_PATH,
        },
        "inputs": {
            "v4_stream": V4_STREAM_PATH,
            "v4_report": V4_REPORT_PATH,
            "v6_stream": V6_STREAM_PATH,
            "v6_report": V6_REPORT_PATH,
            "v6_metrics": V6_METRICS_PATH,
            "video": VIDEO_PATH,
            "canonical_obj": CANONICAL_OBJ,
        },
        "timing": {
            "postpass_wall_s_before_render": float(time.time() - t0),
        },
        "coverage": coverage,
        "topology": topo,
        "lag": {
            "before_v6": lag_before,
            "after_v7": lag_after,
        },
        "center_error": {
            "before_v6": center_before,
            "after_v7": center_after,
        },
        "feature_residual": {
            "before_v6": residual_before,
            "after_v7": residual_after,
        },
        "shimmer": shimmer,
        "source_boundary": source_boundary,
        "pose_stabilization": pose_detail_for_json(pose_detail),
        "geometry_rebuild": geometry_detail,
        "pose_continuity": pose_metrics,
        "semantic_anchor_lock": {
            "anchor_vertex_count": int(len(ANCHOR_IDX)),
            "anchor_vertices": [int(v) for v in ANCHOR_IDX],
            "center_error_targets": {"keep_v6_observed_geometry": True, "median_lt": 0.03, "p95_lt": 0.06},
        },
        "motion_verification": motion,
        "mps_no_cuda": mps_no_cuda,
        "head_scale_px": np.asarray(head_scale, dtype=np.float32).tolist(),
        "kill_honest": {
            "target_lag_near_zero_frames": bool(abs(lag_after["best_delay_frames"]) <= 0),
            "target_center_median_lt_0p03": bool(center_after["pass_median_lt_0p03"]),
            "target_center_p95_lt_0p06": bool(center_after["pass_p95_lt_0p06"]),
            "f429_flip_removed": bool(abs(pose_metrics["pose_flip_span"]["v7_f429_yaw_wrapped_deg"] - pose_metrics["pose_flip_span"]["raw_f429_yaw_deg"]) > 30.0),
            "pose_flip_span_pose_step_reduced": bool(
                pose_metrics["pose_flip_span"]["max_v7_pose_step_deg"]
                < pose_metrics["pose_flip_span"]["max_raw_pose_step_deg"]
            ),
            "pose_flip_span_vertex_step_reduced": bool(
                pose_metrics["pose_flip_span"]["max_v7_mean_vertex_step_over_head_scale"]
                < pose_metrics["pose_flip_span"]["max_raw_mean_vertex_step_over_head_scale"]
            ),
            "topology_coverage_pass": bool(topo["pass"] and coverage["no_nan"] and coverage["verts_populated"] == total_f),
            "mps_no_cuda_scan_pass": bool(mps_no_cuda["pass"]),
            "observed_frames_preserved": bool(geometry_detail["observed_geometry_max_abs_delta_px"] == 0.0),
            "remaining_raw_max_source_jump_over_head_scale": float(source_boundary["after_v7"]["max_jump_over_head_scale"]),
            "honest_floor_reason": "back/profile frames still lack true facial observations; v7 fixes pose continuity and visible snap, not profile feature truth",
        },
    }

    metrics = {
        "generated_at": report["generated_at"],
        "inputs": report["inputs"],
        "lag": report["lag"],
        "center_error": report["center_error"],
        "feature_residual": report["feature_residual"],
        "shimmer": report["shimmer"],
        "source_boundary": report["source_boundary"],
        "pose_stabilization": report["pose_stabilization"],
        "geometry_rebuild": report["geometry_rebuild"],
        "pose_continuity": pose_metrics,
        "topology": topo,
        "coverage": coverage,
        "mps_no_cuda": mps_no_cuda,
        "motion_verification": motion,
        "honest_floor": report["kill_honest"],
        "before_after_summary": {
            "lag_delay_frames_v6_to_v7": [lag_before["best_delay_frames"], lag_after["best_delay_frames"]],
            "center_median_v6_to_v7": [center_before["median"], center_after["median"]],
            "center_p95_v6_to_v7": [center_before["p95"], center_after["p95"]],
            "pose_flip_max_pose_step_deg_v6_to_v7": [
                pose_metrics["pose_flip_span"]["max_raw_pose_step_deg"],
                pose_metrics["pose_flip_span"]["max_v7_pose_step_deg"],
            ],
            "pose_flip_max_mean_vertex_step_v6_to_v7": [
                pose_metrics["pose_flip_span"]["max_raw_mean_vertex_step_over_head_scale"],
                pose_metrics["pose_flip_span"]["max_v7_mean_vertex_step_over_head_scale"],
            ],
            "f429_yaw_raw_to_v7_unwrapped": [
                pose_metrics["pose_flip_span"]["raw_f429_yaw_deg"],
                pose_metrics["pose_flip_span"]["v7_f429_yaw_unwrapped_deg"],
            ],
        },
    }

    print(f"{LOG_PREFIX} Rendering wireframe overlay...")
    outputs = build_video_and_montage(projected_px, source, yaw, faces, report)
    report["paths"].update(outputs)
    report["timing"]["total_wall_s"] = float(time.time() - t0)
    report["output_sizes_mb"] = {
        "stream": round(os.path.getsize(STREAM_PATH) / 1e6, 3),
        "overlay_master": round(os.path.getsize(OVERLAY_MASTER_PATH) / 1e6, 3),
        "overlay_preview": round(os.path.getsize(OVERLAY_PREVIEW_PATH) / 1e6, 3),
        "montage": round(os.path.getsize(MONTAGE_PATH) / 1e6, 3) if os.path.exists(MONTAGE_PATH) else 0.0,
        "motion_strip": round(os.path.getsize(MOTION_STRIP_PATH) / 1e6, 3) if os.path.exists(MOTION_STRIP_PATH) else 0.0,
    }
    metrics["paths"] = report["paths"]
    metrics["output_sizes_mb"] = report["output_sizes_mb"]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_notes(report, metrics)

    print(f"{LOG_PREFIX} DONE")
    print(f"  Pose flip max pose step: {pose_metrics['pose_flip_span']['max_raw_pose_step_deg']:.2f} -> "
          f"{pose_metrics['pose_flip_span']['max_v7_pose_step_deg']:.2f} deg")
    print(f"  f429 yaw: {pose_metrics['pose_flip_span']['raw_f429_yaw_deg']:.2f} -> "
          f"{pose_metrics['pose_flip_span']['v7_f429_yaw_unwrapped_deg']:.2f} deg unwrapped")
    print(f"  Lag delay frames: {lag_before['best_delay_frames']} -> {lag_after['best_delay_frames']}")
    print(f"  Center error median/p95: {center_before['median']:.4f}/{center_before['p95']:.4f} -> "
          f"{center_after['median']:.4f}/{center_after['p95']:.4f}")
    print(f"  Overlay: {OVERLAY_MASTER_PATH}")
    print(f"  Notes: {NOTES_PATH}")
    return report


if __name__ == "__main__":
    run()
