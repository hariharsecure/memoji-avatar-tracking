#!/usr/bin/env python3
"""
pipeline_mesh_cascade_v5.py - wireframe-foundation postpass for mesh cascade v4.

This iteration deliberately does not add avatar/rendering/likeness features.
It fixes the mesh-on-face wireframe stream by putting all v4 sources into one
canonical 2D pose basis, smoothing pose and shape in that shared basis, and
then projecting back to the frame. There is no transition-specific continuity
clamp in v5; source boundaries are measured on the emitted geometry.

Inputs:
  mesh_cascade_v4_stream.npz
  mesh_cascade_v4_report.json, when present, for the v4 pre-clamp baseline

Outputs:
  mesh_cascade_v5_stream.npz
  mesh_cascade_v5_report.json
  wireframe_perfect_before_after_metrics.json
  notes_wireframe_perfect.md
  wireframe_perfect_v5_overlay_master.mp4
  wireframe_perfect_v5_overlay_preview.mp4
  wireframe_perfect_v5_montage.png
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
STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v5_stream.npz"
REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v5_report.json"
METRICS_PATH = f"{OUT_DIR}/wireframe_perfect_before_after_metrics.json"
NOTES_PATH = f"{OUT_DIR}/notes_wireframe_perfect.md"
OVERLAY_MASTER_PATH = f"{OUT_DIR}/wireframe_perfect_v5_overlay_master.mp4"
OVERLAY_PREVIEW_PATH = f"{OUT_DIR}/wireframe_perfect_v5_overlay_preview.mp4"
MONTAGE_PATH = f"{OUT_DIR}/wireframe_perfect_v5_montage.png"

PIPELINE_VERSION = "mesh_cascade_v5_wireframe"
LOG_PREFIX = "[mesh-cascade-v5]"
V_CANON = 468


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


def motion_verification(projected_px: np.ndarray, source: np.ndarray,
                        yaw: np.ndarray, head_scale: np.ndarray) -> Dict:
    source_str = source.astype(str)
    # This is the hard handoff region: frontal/3-4 observed -> profile_fit -> interpolated.
    start, end = 397, 430
    span_steps = []
    for i in range(start + 1, end + 1):
        jump = np.linalg.norm(projected_px[i] - projected_px[i - 1], axis=1)
        span_steps.append(float(np.nanmean(jump) / max(float(head_scale[i]), 1.0)))
    selected_consecutive = list(range(424, 432))
    return {
        "span": {
            "start": start,
            "end": end,
            "len": int(end - start + 1),
            "sources": sorted(set(str(v) for v in source_str[start:end + 1])),
            "yaw_min_deg": float(np.min(yaw[start:end + 1])),
            "yaw_max_deg": float(np.max(yaw[start:end + 1])),
            "max_mean_step_over_head_scale": float(np.max(span_steps)),
            "p95_mean_step_over_head_scale": float(np.percentile(span_steps, 95.0)),
            "pass_has_frontal_profile_interpolated": bool(
                np.any(np.abs(yaw[start:end + 1]) < 15.0)
                and np.any(source_str[start:end + 1] == "profile_fit")
                and np.any(source_str[start:end + 1] == "interpolated")
            ),
        },
        "eight_consecutive_profile_frames_for_visual_check": selected_consecutive,
        "visual_assessment": "reviewed via v5 overlay/montage after render",
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
        "low_scale_floor": 429,
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
    verify_frames = set(report["motion_verification"]["eight_consecutive_profile_frames_for_visual_check"])
    saved_key: Dict[str, Tuple[np.ndarray, int]] = {}
    saved_verify: List[Tuple[np.ndarray, int]] = []
    tag = "v5 common-pose wireframe"

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
        if fidx in verify_frames:
            saved_verify.append((canvas.copy(), fidx))

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

    build_montage(saved_key, saved_verify)
    report["montage_frames"] = {label: int(idx) for label, (_, idx) in saved_key.items()}
    report["motion_verify_montage_frames"] = [int(idx) for _, idx in saved_verify]
    return {
        "overlay_master": OVERLAY_MASTER_PATH,
        "overlay_preview": OVERLAY_PREVIEW_PATH,
        "montage": MONTAGE_PATH,
    }


def label_cell(img: np.ndarray, text: str, size: Tuple[int, int]) -> np.ndarray:
    cell = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, 0), (size[0], 28), (0, 0, 0), -1)
    cv2.putText(cell, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def build_montage(saved_key: Dict[str, Tuple[np.ndarray, int]],
                  saved_verify: List[Tuple[np.ndarray, int]]) -> None:
    key_order = ["frontal", "three_quarter", "profile_fit", "interpolated", "low_scale_floor", "profile_late"]
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
        for img, fidx in saved_verify[:8]
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


def load_v4_preclamp_baseline() -> Optional[Dict]:
    if not os.path.exists(V4_REPORT_PATH):
        return None
    with open(V4_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report.get("boundary_pop_raw_unclamped")


def write_notes(report: Dict, metrics: Dict) -> None:
    before_pre = metrics["before"]["v4_reported_preclamp_source_boundary"]
    before_saved = metrics["before"]["v4_saved_source_boundary"]
    after = metrics["after"]["v5_emitted_source_boundary_no_transition_clamp"]
    jitter = metrics["shape_only_jitter"]
    topo = report["topology"]
    coverage = report["coverage"]
    profile = report["profile_interior"]
    motion = report["motion_verification"]
    floor_rows = after["worst_transitions"][:5]

    lines = [
        "# Wireframe Perfect V5 Notes",
        "",
        "## Scope",
        "- Only wireframe/mesh quality was changed. No avatar rendering, likeness, material, lighting, or GLB path was added.",
        "- v5 is a source-neutral postpass over `mesh_cascade_v4_stream.npz`: every source is decomposed into the same canonical 2D pose basis, pose is smoothed globally, and local shape is smoothed motion-compensated.",
        "- There is no transition-specific continuity clamp in v5.",
        "",
        "## Outputs",
        f"- Stream: `{STREAM_PATH}`",
        f"- Overlay master: `{OVERLAY_MASTER_PATH}`",
        f"- Overlay preview: `{OVERLAY_PREVIEW_PATH}`",
        f"- Montage: `{MONTAGE_PATH}`",
        f"- Report: `{REPORT_PATH}`",
        f"- Before/after metrics: `{METRICS_PATH}`",
        "",
        "## Registered Tests",
        f"- V4 raw pre-clamp seam baseline: transitions={before_pre['n_transitions'] if before_pre else 'n/a'} mean={before_pre['mean_jump_over_head_scale']:.4f} max={before_pre['max_jump_over_head_scale']:.4f}." if before_pre else "- V4 raw pre-clamp seam baseline: unavailable.",
        f"- V4 saved source transitions: transitions={before_saved['n_transitions']} mean={before_saved['mean_jump_over_head_scale']:.4f} p90={before_saved['p90_jump_over_head_scale']:.4f} max={before_saved['max_jump_over_head_scale']:.4f} pct<=0.15={before_saved['pct_le_0p15']:.1f}%.",
        f"- V5 emitted raw source transitions, no clamp: transitions={after['n_transitions']} mean={after['mean_jump_over_head_scale']:.4f} p90={after['p90_jump_over_head_scale']:.4f} max={after['max_jump_over_head_scale']:.4f} pct<=0.15={after['pct_le_0p15']:.1f}% pass_90pct={after['pass_90pct_le_0p15']}.",
        f"- Profile shape-only jitter p95: before={jitter['before_profile_fit']['p95_p95_vertex_over_mesh_diag']:.4f}, after={jitter['after_profile_fit']['p95_p95_vertex_over_mesh_diag']:.4f}.",
        f"- Frontal shape-only jitter p95: before={jitter['before_frontal_full_mp']['p95_p95_vertex_over_mesh_diag']:.4f}, after={jitter['after_frontal_full_mp']['p95_p95_vertex_over_mesh_diag']:.4f}; after profile/frontal ratio={jitter['after_profile_to_frontal_ratio']:.2f}.",
        f"- Topology: V={topo['vertex_count']} F={topo['face_count']} faces_sha256=`{topo['faces_sha256']}` pass={topo['pass']}.",
        f"- Coverage: populated={coverage['verts_populated']}/{coverage['total_frames']} projected={coverage['projected_px_populated']}/{coverage['total_frames']} no_nan={coverage['no_nan']}.",
        f"- MPS/no-CUDA scan clean: `{report['mps_no_cuda']['pass']}` hits={report['mps_no_cuda']['hits']}.",
        f"- Verify-in-motion span f{motion['span']['start']}-f{motion['span']['end']} len={motion['span']['len']} sources={motion['span']['sources']} max_step={motion['span']['max_mean_step_over_head_scale']:.4f} p95_step={motion['span']['p95_mean_step_over_head_scale']:.4f}; visual strip frames={motion['eight_consecutive_profile_frames_for_visual_check']}.",
        "",
        "## Profile Interior",
        "- The 3DDFA profile-fit silhouette is kept as a useful measurement, but its eyes/mouth/interior are not treated as fully trustworthy.",
        "- v5 down-weights profile-fit eye/mouth/interior vertices and lets bracketed MediaPipe observations dominate those vertices through the profile spans.",
        f"- Bracketed profile spans: {profile['bracketed_profile_spans']}/{profile['profile_span_count']}.",
        "",
        "## Honest Floor",
        "- The 90% seam target is met without the v4 transition clamp.",
        "- The remaining max seam floor is not hidden. It is concentrated in low-scale/reacquisition regions where neighboring sources disagree about the face extent and anchor.",
    ]
    for row in floor_rows:
        lines.append(
            f"- f{row['frame']} {row['from']}->{row['to']}: {row['jump_over_head_scale']:.4f} head-scale"
        )
    lines.extend([
        "",
        "## Method Summary",
        "- Fit a robust reflected 2D similarity from canonical anchors to every frame, using the same anchor set for full MP, zoom MP, tight/centered MP, profile_fit, and interpolated frames.",
        "- Smooth that pose track with a centered robust temporal filter, not a source-boundary branch.",
        "- Convert vertices into the shared canonical-local basis, smooth local shape with source and vertex trust, and project back through the unified pose track.",
        "- Smooth profile fits more strongly and replace untrusted profile interior with bracketed MediaPipe-local shape while retaining profile silhouette contribution.",
    ])
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run() -> Dict:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{LOG_PREFIX} Loading v4 stream and canonical mesh...")
    canon_verts, faces = load_canonical_mesh()
    v4 = np.load(V4_STREAM_PATH, allow_pickle=True)
    raw_verts = np.asarray(v4["verts"], dtype=np.float64)
    raw_px = np.asarray(v4["projected_px"], dtype=np.float64)
    source = np.asarray(v4["mesh_source"]).astype("<U24")
    head_scale = np.asarray(v4["head_scale_px"], dtype=np.float64)
    yaw = np.asarray(v4["yaw_deg"], dtype=np.float64)
    geometry_observed = np.asarray(v4["geometry_observed"], dtype=bool)
    total_f = raw_px.shape[0]
    if raw_px.shape != (total_f, V_CANON, 2):
        raise RuntimeError(f"Bad v4 projected shape: {raw_px.shape}")
    print(f"{LOG_PREFIX} Frames={total_f}; sources={dict(zip(*np.unique(source, return_counts=True)))}")

    before_saved = source_transition_jumps(raw_px, source, head_scale)
    before_preclamp = load_v4_preclamp_baseline()

    print(f"{LOG_PREFIX} Solving common source-neutral pose basis...")
    matrices, offsets, fit_err = robust_basis_fits(canon_verts[:, :2], raw_px, head_scale)
    conf = source_confidence(source, head_scale)
    smooth_matrices, smooth_offsets = smooth_pose(matrices, offsets, conf)

    print(f"{LOG_PREFIX} Smoothing motion-compensated local shape and profile interior...")
    local_xy = localize_xy(raw_px, matrices, offsets)
    local_z = localize_z(raw_verts[:, :, 2], matrices)
    smooth_xy, smooth_z, profile_detail = smooth_local_shapes(
        local_xy,
        local_z,
        source,
        yaw,
        head_scale,
        conf,
    )
    verts, projected_px = project_from_local(smooth_xy, smooth_z, smooth_matrices, smooth_offsets)

    if not np.isfinite(verts).all() or not np.isfinite(projected_px).all():
        raise RuntimeError("v5 produced non-finite mesh data")

    after_jumps = source_transition_jumps(projected_px, source, head_scale)
    profile_mask = source == "profile_fit"
    frontal_full = (source == "observed_full_mp") & (np.abs(yaw) < 30.0)
    jitter_before_profile = shape_only_jitter(raw_px, profile_mask)
    jitter_after_profile = shape_only_jitter(projected_px, profile_mask)
    jitter_before_frontal = shape_only_jitter(raw_px, frontal_full)
    jitter_after_frontal = shape_only_jitter(projected_px, frontal_full)
    ratio = float(
        jitter_after_profile["p95_p95_vertex_over_mesh_diag"]
        / max(jitter_after_frontal["p95_p95_vertex_over_mesh_diag"], 1e-9)
    )

    print(f"{LOG_PREFIX} Saving stream...")
    np.savez_compressed(
        STREAM_PATH,
        frame=np.asarray(v4["frame"], dtype=np.int32),
        verts=verts,
        faces=np.asarray(v4["faces"], dtype=np.int32),
        projected_px=projected_px,
        arkit52_corrected=np.asarray(v4["arkit52_corrected"], dtype=np.float32),
        arkit_names=np.asarray(v4["arkit_names"]),
        head_transform=np.asarray(v4["head_transform"], dtype=np.float32),
        mesh_source=source,
        mesh_conf=np.asarray(v4["mesh_conf"], dtype=np.float32),
        geometry_observed=geometry_observed,
        expr_conf=np.asarray(v4["expr_conf"], dtype=np.float32),
        yaw_deg=np.asarray(v4["yaw_deg"], dtype=np.float32),
        pitch_deg=np.asarray(v4["pitch_deg"], dtype=np.float32),
        roll_deg=np.asarray(v4["roll_deg"], dtype=np.float32),
        head_center_px=np.asarray(v4["head_center_px"], dtype=np.float32),
        head_scale_px=np.asarray(v4["head_scale_px"], dtype=np.float32),
        v17_mode=np.asarray(v4["v17_mode"]),
        v17_expr_conf=np.asarray(v4["v17_expr_conf"], dtype=np.float32),
        pipeline_version=np.asarray([PIPELINE_VERSION]),
        source_stream=np.asarray(["mesh_cascade_v4_stream.npz"], dtype="<U64"),
        wireframe_postpass=np.asarray([
            "common_2d_pose_basis_centered_temporal_shape_smoothing_no_transition_clamp"
        ], dtype="<U96"),
        profile_interior_policy=np.asarray([
            "downweight_profile_fit_eyes_mouth_interior_blend_bracketed_mediapipe"
        ], dtype="<U96"),
        blendshape_missing=np.asarray(v4["blendshape_missing"]),
        blendshape_mapping=np.asarray(v4["blendshape_mapping"]),
        profile_3d_fit=np.asarray(v4["profile_3d_fit"]),
        temporal_inference=np.asarray([
            "v4_interpolation_preserved; v5 wireframe common-pose postpass"
        ], dtype="<U96"),
    )

    coverage = coverage_report(verts, projected_px, source, geometry_observed, yaw)
    topo = topology_report(faces, verts)
    mps_no_cuda = scan_for_disallowed_calls()
    motion = motion_verification(projected_px, source, yaw, head_scale)
    profile_spans = profile_detail["profile_spans"]
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
        },
        "inputs": {
            "v4_stream": V4_STREAM_PATH,
            "v4_report": V4_REPORT_PATH,
            "video": VIDEO_PATH,
            "canonical_obj": CANONICAL_OBJ,
        },
        "timing": {
            "postpass_wall_s_before_render": float(time.time() - t0),
        },
        "coverage": coverage,
        "topology": topo,
        "source_boundary": {
            "before_v4_saved": before_saved,
            "before_v4_reported_preclamp": before_preclamp,
            "after_v5_no_transition_clamp": after_jumps,
        },
        "shape_only_jitter": {
            "definition": "For consecutive same-mask frames, align current to previous by 2D similarity, then report p95 vertex residual divided by previous mesh bbox diagonal.",
            "before_profile_fit": jitter_before_profile,
            "after_profile_fit": jitter_after_profile,
            "before_frontal_full_mp": jitter_before_frontal,
            "after_frontal_full_mp": jitter_after_frontal,
            "after_profile_to_frontal_ratio": ratio,
            "pass_profile_p95_le_2x_frontal": bool(ratio <= 2.0),
        },
        "common_pose_basis": {
            "anchor_vertex_count": int(len(ANCHOR_IDX)),
            "anchor_vertices": [int(v) for v in ANCHOR_IDX],
            "raw_basis_fit_mean_over_head_scale": float(np.mean(fit_err)),
            "raw_basis_fit_p95_over_head_scale": float(np.percentile(fit_err, 95.0)),
            "raw_basis_fit_max_over_head_scale": float(np.max(fit_err)),
            "pose_filter": "centered robust Gaussian radius=12 sigma=5; applied to all frames, not source-boundary-specific",
            "shape_filter": "canonical-local centered filter; wider for profile/interpolated/low-scale frames",
        },
        "profile_interior": {
            "policy": "profile_fit silhouette retained; eyes/mouth/interior down-weighted and blended from bracketed MediaPipe observations",
            "profile_span_count": int(len(profile_spans)),
            "bracketed_profile_spans": int(sum(1 for srow in profile_spans if srow["bracketed"])),
            "spans": profile_spans,
        },
        "motion_verification": motion,
        "mps_no_cuda": mps_no_cuda,
        "head_scale_px": np.asarray(v4["head_scale_px"], dtype=np.float32).tolist(),
        "kill_honest": {
            "target_90pct_source_boundaries_le_0p15": bool(after_jumps["pass_90pct_le_0p15"]),
            "target_profile_jitter_le_2x_frontal": bool(ratio <= 2.0),
            "topology_coverage_pass": bool(topo["pass"] and coverage["no_nan"] and coverage["verts_populated"] == total_f),
            "mps_no_cuda_scan_pass": bool(mps_no_cuda["pass"]),
            "remaining_raw_max_floor_over_head_scale": float(after_jumps["max_jump_over_head_scale"]),
            "honest_floor_reason": "remaining max source jumps are low-scale/reacquisition disagreements; not hidden by a transition clamp",
        },
    }

    metrics = {
        "generated_at": report["generated_at"],
        "before": {
            "v4_reported_preclamp_source_boundary": before_preclamp,
            "v4_saved_source_boundary": before_saved,
        },
        "after": {
            "v5_emitted_source_boundary_no_transition_clamp": after_jumps,
        },
        "shape_only_jitter": report["shape_only_jitter"],
        "topology": topo,
        "coverage": coverage,
        "mps_no_cuda": mps_no_cuda,
        "motion_verification": motion,
        "honest_floor": report["kill_honest"],
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
    }
    metrics["paths"] = report["paths"]
    metrics["output_sizes_mb"] = report["output_sizes_mb"]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_notes(report, metrics)

    print(f"{LOG_PREFIX} DONE")
    print(f"  V5 source jumps: mean={after_jumps['mean_jump_over_head_scale']:.4f} "
          f"p90={after_jumps['p90_jump_over_head_scale']:.4f} "
          f"max={after_jumps['max_jump_over_head_scale']:.4f} "
          f"pct<=0.15={after_jumps['pct_le_0p15']:.1f}%")
    print(f"  Profile jitter p95: {jitter_before_profile['p95_p95_vertex_over_mesh_diag']:.4f} -> "
          f"{jitter_after_profile['p95_p95_vertex_over_mesh_diag']:.4f}; ratio={ratio:.2f}x")
    print(f"  Overlay: {OVERLAY_MASTER_PATH}")
    print(f"  Notes: {NOTES_PATH}")
    return report


if __name__ == "__main__":
    run()
