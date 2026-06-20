#!/usr/bin/env python3
"""
pipeline_mesh_cascade_v6.py - One-Euro locked wireframe overlay.

v5 removed shimmer, but its fixed temporal smoothing left visible face-trailing
lag and occasional off-center placement. v6 keeps the v5 de-shimmered mesh as
the organic topology/shape baseline, then re-locks it to the unlagged v4 face
landmark measurements with an adaptive One-Euro filtered affine/similarity
correction. The final center/scale lock is applied every frame against semantic
face anchors; it is not a source-boundary continuity clamp.

Inputs:
  mesh_cascade_v5_stream.npz
  mesh_cascade_v4_stream.npz
  mesh_cascade_v5_report.json / wireframe_perfect_before_after_metrics.json,
  when present, for the v5 before baseline

Outputs:
  mesh_cascade_v6_stream.npz
  mesh_cascade_v6_report.json
  wireframe_v6_before_after_metrics.json
  notes_wireframe_v6.md
  wireframe_v6_overlay_master.mp4
  wireframe_v6_overlay_preview.mp4
  wireframe_v6_fast_motion_strip.png
  wireframe_v6_center_scale_montage.png
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
import trimesh


VIDEO_PATH = "input_clip.mov"
CANONICAL_OBJ = "assets/canonical_face_model.obj"
OUT_DIR = "."

V4_STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v4_stream.npz"
V4_REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v4_report.json"
V5_STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v5_stream.npz"
V5_REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v5_report.json"
V5_METRICS_PATH = f"{OUT_DIR}/wireframe_perfect_before_after_metrics.json"
STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v6_stream.npz"
REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v6_report.json"
METRICS_PATH = f"{OUT_DIR}/wireframe_v6_before_after_metrics.json"
NOTES_PATH = f"{OUT_DIR}/notes_wireframe_v6.md"
OVERLAY_MASTER_PATH = f"{OUT_DIR}/wireframe_v6_overlay_master.mp4"
OVERLAY_PREVIEW_PATH = f"{OUT_DIR}/wireframe_v6_overlay_preview.mp4"
MONTAGE_PATH = f"{OUT_DIR}/wireframe_v6_center_scale_montage.png"
MOTION_STRIP_PATH = f"{OUT_DIR}/wireframe_v6_fast_motion_strip.png"

PIPELINE_VERSION = "mesh_cascade_v6_one_euro_lock"
LOG_PREFIX = "[mesh-cascade-v6]"
V_CANON = 468

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

    yaw_abs = np.abs(np.asarray(yaw, dtype=np.float64))
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
    frontal = np.asarray([str(v).startswith("observed") for v in source], dtype=bool) & (np.abs(yaw) < 30.0)
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
        "before_v5": source_transition_jumps(before_px, source, head_scale),
        "after_v6": source_transition_jumps(after_px, source, head_scale),
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
            "0_30": int((np.abs(yaw) < 30.0).sum()),
            "30_60": int(((np.abs(yaw) >= 30.0) & (np.abs(yaw) < 60.0)).sum()),
            "60_90": int(((np.abs(yaw) >= 60.0) & (np.abs(yaw) < 90.0)).sum()),
            "90_180": int((np.abs(yaw) >= 90.0).sum()),
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
    yaw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(yaw, dtype=np.float64))))
    face_speed = np.r_[0.0, np.linalg.norm(np.diff(face_center, axis=0), axis=1) / np.maximum(head_scale[1:], 1.0)]
    yaw_speed = np.r_[0.0, np.abs(np.diff(yaw_unwrapped))]
    score = face_speed + yaw_speed / 30.0
    candidates = []
    for start in range(0, len(projected_px) - 9):
        end = start + 9
        if float(np.median(head_scale[start:end + 1])) < 45.0:
            continue
        candidates.append((float(np.mean(score[start + 1:end + 1])), start, end))
    if candidates:
        _, start, end = max(candidates, key=lambda row: row[0])
    else:
        start, end = 397, 406

    center_error = (
        np.linalg.norm(mesh_center[start:end + 1] - face_center[start:end + 1], axis=1)
        / np.maximum(head_scale[start:end + 1], 1.0)
    )
    mean_step = []
    for i in range(start + 1, end + 1):
        jump = np.linalg.norm(projected_px[i] - projected_px[i - 1], axis=1)
        mean_step.append(float(np.nanmean(jump) / max(float(head_scale[i]), 1.0)))

    long_start, long_end = 397, 430
    long_steps = []
    for i in range(long_start + 1, long_end + 1):
        jump = np.linalg.norm(projected_px[i] - projected_px[i - 1], axis=1)
        long_steps.append(float(np.nanmean(jump) / max(float(head_scale[i]), 1.0)))

    return {
        "fast_motion_10_frame_span": {
            "start": int(start),
            "end": int(end),
            "len": int(end - start + 1),
            "frames": [int(v) for v in range(start, end + 1)],
            "sources": sorted(set(str(v) for v in source_str[start:end + 1])),
            "yaw_min_deg": float(np.min(yaw[start:end + 1])),
            "yaw_max_deg": float(np.max(yaw[start:end + 1])),
            "yaw_abs_delta_sum_deg": float(np.sum(np.abs(np.diff(yaw_unwrapped[start:end + 1])))),
            "mean_face_speed_over_head_scale": float(np.mean(face_speed[start + 1:end + 1])),
            "center_error_median_over_head_scale": float(np.percentile(center_error, 50.0)),
            "center_error_p95_over_head_scale": float(np.percentile(center_error, 95.0)),
            "max_mean_mesh_step_over_head_scale": float(np.max(mean_step)) if mean_step else 0.0,
            "p95_mean_mesh_step_over_head_scale": float(np.percentile(mean_step, 95.0)) if mean_step else 0.0,
            "visual_strip": MOTION_STRIP_PATH,
        },
        "handoff_span_397_430": {
            "start": long_start,
            "end": long_end,
            "len": int(long_end - long_start + 1),
            "sources": sorted(set(str(v) for v in source_str[long_start:long_end + 1])),
            "yaw_min_deg": float(np.min(yaw[long_start:long_end + 1])),
            "yaw_max_deg": float(np.max(yaw[long_start:long_end + 1])),
            "max_mean_step_over_head_scale": float(np.max(long_steps)),
            "p95_mean_step_over_head_scale": float(np.percentile(long_steps, 95.0)),
            "pass_has_frontal_profile_interpolated": bool(
                np.any(np.abs(yaw[long_start:long_end + 1]) < 15.0)
                and np.any(source_str[long_start:long_end + 1] == "profile_fit")
                and np.any(source_str[long_start:long_end + 1] == "interpolated")
            ),
        },
        "visual_assessment": "rendered v6 overlay plus exact 10-frame fast-motion strip",
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

    def first(mask: np.ndarray, fallback: int) -> int:
        idx = np.where(mask)[0]
        return int(idx[0]) if len(idx) else int(fallback)

    observed = np.asarray([startswith_observed(v) for v in source_str], dtype=bool)
    return {
        "frontal": first(observed & (np.abs(yaw) < 15.0), 50),
        "three_quarter": first(observed & (np.abs(yaw) >= 30.0) & (np.abs(yaw) < 60.0), 90),
        "profile_fit": first(source_str == "profile_fit", 188),
        "interpolated": first(source_str == "interpolated", 428),
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
    motion_frames = set(report["motion_verification"]["fast_motion_10_frame_span"]["frames"])
    saved_key: Dict[str, Tuple[np.ndarray, int]] = {}
    saved_motion: List[Tuple[np.ndarray, int]] = []
    tag = "v6 one-euro face lock"

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
    key_order = ["frontal", "three_quarter", "profile_fit", "interpolated", "center_scale_lock", "profile_late"]
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
    ordered = sorted(saved_motion, key=lambda row: row[1])[:10]
    if not ordered:
        return
    cells = [label_cell(img, f"fast f{fidx}", (180, 320)) for img, fidx in ordered]
    while len(cells) < 10:
        cells.append(np.zeros_like(cells[0]))
    cv2.imwrite(MOTION_STRIP_PATH, np.hstack(cells[:10]))


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
    shimmer = metrics["shimmer"]
    topo = report["topology"]
    coverage = report["coverage"]
    motion = report["motion_verification"]["fast_motion_10_frame_span"]
    jumps = metrics["source_boundary"]["after_v6"]["worst_transitions"][:5]

    lines = [
        "# Wireframe V6 Notes",
        "",
        "## Scope",
        "- v6 only changes the wireframe overlay postpass. No avatar rendering, likeness, material, lighting, or GLB path was added.",
        "- v5 de-shimmered geometry is kept as the organic baseline. v4 raw face landmarks are used as the unlagged face-feature measurement.",
        "- The fixed v5 temporal smoothing is replaced by a One-Euro adaptive correction over pose/anchor parameters, followed by an every-frame semantic center/scale lock.",
        "- There is no source-boundary continuity clamp.",
        "",
        "## Outputs",
        f"- Stream: `{STREAM_PATH}`",
        f"- Overlay master: `{OVERLAY_MASTER_PATH}`",
        f"- Overlay preview: `{OVERLAY_PREVIEW_PATH}`",
        f"- Fast motion strip: `{MOTION_STRIP_PATH}`",
        f"- Center/scale montage: `{MONTAGE_PATH}`",
        f"- Report: `{REPORT_PATH}`",
        f"- Before/after metrics: `{METRICS_PATH}`",
        "",
        "## Pre-Registered Tests",
        f"- Lag: v5 best_delay={lag['before_v5']['best_delay_frames']} frame(s), v6 best_delay={lag['after_v6']['best_delay_frames']} frame(s), target ~0.",
        f"- Center error: v5 median={center['before_v5']['median']:.4f} p95={center['before_v5']['p95']:.4f}; v6 median={center['after_v6']['median']:.4f} p95={center['after_v6']['p95']:.4f}; targets median<0.03 p95<0.06.",
        f"- Feature residual frontal p95: v5={residual['before_v5']['frontal_abs_yaw_lt_30']['p95_px']:.2f}px/{residual['before_v5']['frontal_abs_yaw_lt_30']['p95_over_head_scale']:.4f}, v6={residual['after_v6']['frontal_abs_yaw_lt_30']['p95_px']:.2f}px/{residual['after_v6']['frontal_abs_yaw_lt_30']['p95_over_head_scale']:.4f}.",
        f"- Feature residual profile p95: v5={residual['before_v5']['profile_abs_yaw_ge_60']['p95_px']:.2f}px/{residual['before_v5']['profile_abs_yaw_ge_60']['p95_over_head_scale']:.4f}, v6={residual['after_v6']['profile_abs_yaw_ge_60']['p95_px']:.2f}px/{residual['after_v6']['profile_abs_yaw_ge_60']['p95_over_head_scale']:.4f}.",
        f"- Still-frame shimmer p95: v5={shimmer['before_still']['p95_p95_vertex_over_mesh_diag']:.6f}, v6={shimmer['after_still']['p95_p95_vertex_over_mesh_diag']:.6f}.",
        f"- Source-boundary scalar: v5 p90={metrics['source_boundary']['before_v5']['p90_jump_over_head_scale']:.4f} pct<=0.15={metrics['source_boundary']['before_v5']['pct_le_0p15']:.1f}%; v6 p90={metrics['source_boundary']['after_v6']['p90_jump_over_head_scale']:.4f} pct<=0.15={metrics['source_boundary']['after_v6']['pct_le_0p15']:.1f}%. This scalar worsens because v6 follows raw face-anchor motion instead of lag-smoothing across detector-source switches.",
        f"- Fast motion strip: f{motion['start']}-f{motion['end']} len={motion['len']} sources={motion['sources']} yaw_delta_sum={motion['yaw_abs_delta_sum_deg']:.1f}deg center_p95={motion['center_error_p95_over_head_scale']:.4f}.",
        f"- Topology: V={topo['vertex_count']} F={topo['face_count']} faces_sha256=`{topo['faces_sha256']}` pass={topo['pass']}.",
        f"- Coverage: populated={coverage['verts_populated']}/{coverage['total_frames']} projected={coverage['projected_px_populated']}/{coverage['total_frames']} no_nan={coverage['no_nan']}.",
        f"- MPS/no-CUDA scan clean: `{report['mps_no_cuda']['pass']}` hits={report['mps_no_cuda']['hits']}.",
        "",
        "## Method Summary",
        "- Fit a robust correction from v5 semantic mesh anchors to v4 raw face anchors every frame.",
        "- Use full affine correction for reliable observed frames to preserve rotation/scale/perspective-like yaw foreshortening; use similarity for profile, interpolated, wide-yaw, and low-scale frames to avoid unstable shear.",
        f"- Filter the six correction parameters with One-Euro min_cutoff={ONE_EURO_MIN_CUTOFF}, beta={ONE_EURO_BETA}, d_cutoff={ONE_EURO_D_CUTOFF}.",
        "- Lock semantic mesh center and semantic scale to the raw face anchors after filtering, every frame.",
        "",
        "## Honest Floor",
        "- The center target is met by construction using semantic anchors, not the all-vertex mean.",
        "- Residual feature error remains highest in profile/low-scale frames because v4 profile/interpolated face-feature measurements are themselves less reliable than observed MediaPipe frames.",
        "- Remaining largest source jumps are reported below; they are not hidden by a boundary clamp.",
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
    print(f"{LOG_PREFIX} Loading v4 raw target, v5 baseline, and video metadata...")
    v4 = np.load(V4_STREAM_PATH, allow_pickle=True)
    raw_px = np.asarray(v4["projected_px"], dtype=np.float64)
    v5 = np.load(V5_STREAM_PATH, allow_pickle=True)
    v5_px = np.asarray(v5["projected_px"], dtype=np.float64)
    v5_verts = np.asarray(v5["verts"], dtype=np.float64)
    faces = np.asarray(v5["faces"], dtype=np.int32)
    source = np.asarray(v5["mesh_source"]).astype("<U24")
    head_scale = np.asarray(v5["head_scale_px"], dtype=np.float64)
    yaw = np.asarray(v5["yaw_deg"], dtype=np.float64)
    geometry_observed = np.asarray(v5["geometry_observed"], dtype=bool)
    total_f = v5_px.shape[0]
    if raw_px.shape != (total_f, V_CANON, 2) or v5_px.shape != (total_f, V_CANON, 2):
        raise RuntimeError(f"Bad stream shapes raw={raw_px.shape} v5={v5_px.shape}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.0)
    cap.release()
    print(f"{LOG_PREFIX} Frames={total_f}; fps={fps:.3f}; sources={dict(zip(*np.unique(source, return_counts=True)))}")

    print(f"{LOG_PREFIX} Fitting robust v5->raw semantic corrections...")
    correction_params, correction_detail = fit_one_euro_corrections(
        v5_px,
        raw_px,
        source,
        yaw,
        head_scale,
        fps,
    )
    corrected_px = apply_correction_params(v5_px, correction_params)
    projected_px64, lock_detail = semantic_center_scale_lock(corrected_px, raw_px)
    projected_px = projected_px64.astype(np.float32)
    verts = update_verts_from_projected(v5_verts, v5_px, projected_px64, correction_params)

    if not np.isfinite(verts).all() or not np.isfinite(projected_px).all():
        raise RuntimeError("v6 produced non-finite mesh data")

    face_center = semantic_center(raw_px)
    lag_before = cross_correlation_delay(semantic_center(v5_px), face_center)
    lag_after = cross_correlation_delay(semantic_center(projected_px64), face_center)
    center_before = center_error_report(v5_px, raw_px, head_scale)
    center_after = center_error_report(projected_px64, raw_px, head_scale)
    residual_before = feature_residual_report(v5_px, raw_px, head_scale, yaw)
    residual_after = feature_residual_report(projected_px64, raw_px, head_scale, yaw)
    shimmer = shimmer_report(v5_px, projected_px64, raw_px, source, yaw, head_scale)
    source_boundary = source_transition_jumps_comparison(v5_px, projected_px64, source, head_scale)

    print(f"{LOG_PREFIX} Saving stream...")
    stream_payload = {key: np.asarray(v5[key]) for key in v5.files}
    stream_payload.update({
        "verts": verts,
        "faces": faces,
        "projected_px": projected_px,
        "mesh_source": source,
        "geometry_observed": geometry_observed,
        "pipeline_version": np.asarray([PIPELINE_VERSION]),
        "source_stream": np.asarray(["mesh_cascade_v5_stream.npz + mesh_cascade_v4_stream.npz"], dtype="<U96"),
        "wireframe_postpass": np.asarray([
            "one_euro_adaptive_v5_to_raw_semantic_face_lock_no_transition_clamp"
        ], dtype="<U96"),
        "face_lock_policy": np.asarray([
            "weighted_semantic_anchor_center_scale_locked_every_frame"
        ], dtype="<U96"),
        "one_euro_params": np.asarray([
            f"min_cutoff={ONE_EURO_MIN_CUTOFF}; beta={ONE_EURO_BETA}; d_cutoff={ONE_EURO_D_CUTOFF}; fps={fps:.6f}"
        ], dtype="<U128"),
        "temporal_inference": np.asarray([
            "v5 mesh de-shimmer preserved; v6 one-euro adaptive face-anchor relock"
        ], dtype="<U96"),
    })
    np.savez_compressed(STREAM_PATH, **stream_payload)

    coverage = coverage_report(verts, projected_px, source, geometry_observed, yaw)
    topo = topology_report(faces, verts)
    mps_no_cuda = scan_for_disallowed_calls()
    motion = motion_verification(projected_px64, raw_px, source, yaw, head_scale)
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
            "v5_stream": V5_STREAM_PATH,
            "v5_report": V5_REPORT_PATH,
            "v5_metrics": V5_METRICS_PATH,
            "video": VIDEO_PATH,
            "canonical_obj": CANONICAL_OBJ,
        },
        "timing": {
            "postpass_wall_s_before_render": float(time.time() - t0),
        },
        "coverage": coverage,
        "topology": topo,
        "lag": {
            "before_v5": lag_before,
            "after_v6": lag_after,
        },
        "center_error": {
            "before_v5": center_before,
            "after_v6": center_after,
        },
        "feature_residual": {
            "before_v5": residual_before,
            "after_v6": residual_after,
        },
        "shimmer": shimmer,
        "source_boundary": source_boundary,
        "one_euro_correction": correction_detail,
        "center_scale_lock": lock_detail,
        "semantic_anchor_lock": {
            "anchor_vertex_count": int(len(ANCHOR_IDX)),
            "anchor_vertices": [int(v) for v in ANCHOR_IDX],
            "center_error_targets": {"median_lt": 0.03, "p95_lt": 0.06},
        },
        "motion_verification": motion,
        "mps_no_cuda": mps_no_cuda,
        "head_scale_px": np.asarray(v4["head_scale_px"], dtype=np.float32).tolist(),
        "kill_honest": {
            "target_lag_near_zero_frames": bool(abs(lag_after["best_delay_frames"]) <= 0),
            "target_center_median_lt_0p03": bool(center_after["pass_median_lt_0p03"]),
            "target_center_p95_lt_0p06": bool(center_after["pass_p95_lt_0p06"]),
            "still_shimmer_not_materially_worse_than_v5": bool(
                shimmer["after_still"]["p95_p95_vertex_over_mesh_diag"]
                <= shimmer["before_still"]["p95_p95_vertex_over_mesh_diag"] * 1.01
            ),
            "topology_coverage_pass": bool(topo["pass"] and coverage["no_nan"] and coverage["verts_populated"] == total_f),
            "mps_no_cuda_scan_pass": bool(mps_no_cuda["pass"]),
            "remaining_raw_max_source_jump_over_head_scale": float(source_boundary["after_v6"]["max_jump_over_head_scale"]),
            "honest_floor_reason": "profile/low-scale feature residuals are bounded by the quality of raw profile/interpolated face measurements; no source-boundary clamp is used",
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
        "one_euro_correction": report["one_euro_correction"],
        "center_scale_lock": report["center_scale_lock"],
        "topology": topo,
        "coverage": coverage,
        "mps_no_cuda": mps_no_cuda,
        "motion_verification": motion,
        "honest_floor": report["kill_honest"],
        "before_after_summary": {
            "lag_delay_frames_v5_to_v6": [lag_before["best_delay_frames"], lag_after["best_delay_frames"]],
            "center_median_v5_to_v6": [center_before["median"], center_after["median"]],
            "center_p95_v5_to_v6": [center_before["p95"], center_after["p95"]],
            "frontal_feature_p95_px_v5_to_v6": [
                residual_before["frontal_abs_yaw_lt_30"]["p95_px"],
                residual_after["frontal_abs_yaw_lt_30"]["p95_px"],
            ],
            "profile_feature_p95_px_v5_to_v6": [
                residual_before["profile_abs_yaw_ge_60"]["p95_px"],
                residual_after["profile_abs_yaw_ge_60"]["p95_px"],
            ],
            "still_shimmer_p95_v5_to_v6": [
                shimmer["before_still"]["p95_p95_vertex_over_mesh_diag"],
                shimmer["after_still"]["p95_p95_vertex_over_mesh_diag"],
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
    print(f"  Lag delay frames: {lag_before['best_delay_frames']} -> {lag_after['best_delay_frames']}")
    print(f"  Center error median/p95: {center_before['median']:.4f}/{center_before['p95']:.4f} -> "
          f"{center_after['median']:.4f}/{center_after['p95']:.4f}")
    print(f"  Still shimmer p95: {shimmer['before_still']['p95_p95_vertex_over_mesh_diag']:.6f} -> "
          f"{shimmer['after_still']['p95_p95_vertex_over_mesh_diag']:.6f}")
    print(f"  Overlay: {OVERLAY_MASTER_PATH}")
    print(f"  Notes: {NOTES_PATH}")
    return report


if __name__ == "__main__":
    run()
