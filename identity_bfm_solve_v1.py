#!/usr/bin/env python3
"""
identity_bfm_solve_v1.py

Robust BFM identity solve for the subject avatar proof.

Inputs are the existing mesh_cascade_v4 stream and the local 3DDFA_V2 ONNX
checkout. Face boxes are derived from accepted cascade landmarks; FaceBoxes is
not used.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


OUT_DIR = "."
VIDEO_PATH = "input_clip.mov"
STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v4_stream.npz"
THREEDDFA_REPO = f"{OUT_DIR}/_deps/3DDFA_V2"
THREEDDFA_CONFIG = f"{THREEDDFA_REPO}/configs/mb1_120x120.yml"
BFM_PATH = f"{THREEDDFA_REPO}/configs/bfm_noneck_v3.pkl"
TRI_PATH = f"{THREEDDFA_REPO}/configs/tri.pkl"

IDENTITY_PATH = f"{OUT_DIR}/subject_bfm_identity_v1.npz"
REPORT_PATH = f"{OUT_DIR}/subject_bfm_identity_v1_report.json"

PIPELINE_VERSION = "identity_bfm_solve_v1"
LOG_PREFIX = "[identity-bfm-v1]"

MAX_SOLVE_FRAMES = 180
MIN_FRAME_GAP = 4
YAW_GATE_DEG = 20.0
MIN_MESH_CONF = 0.95
MIN_HEAD_SCALE_PX = 40.0
MAX_RESIDUAL_OVER_SCALE = 0.18
MAX_P90_OVER_SCALE = 0.35

# Dlib/3DDFA 68 sparse landmarks mapped to nearby MediaPipe canonical vertices.
DDFA68_TO_MP468 = np.asarray([
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 454,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 197, 4,
    98, 97, 2, 326, 327,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    78, 81, 13, 311, 308, 402, 14, 178,
], dtype=np.int32)


@dataclass
class FitRow:
    frame: int
    yaw_deg: float
    head_scale_px: float
    mesh_conf: float
    mesh_source: str
    box: np.ndarray
    roi_box: np.ndarray
    residual_over_scale: float
    p90_over_scale: float
    alpha_shp: np.ndarray
    alpha_exp: np.ndarray
    param: np.ndarray


def load_tddfa():
    if not os.path.isdir(THREEDDFA_REPO):
        raise RuntimeError(f"missing 3DDFA_V2 repo: {THREEDDFA_REPO}")
    if THREEDDFA_REPO not in sys.path:
        sys.path.insert(0, THREEDDFA_REPO)

    old_cwd = os.getcwd()
    os.chdir(THREEDDFA_REPO)
    try:
        from TDDFA_ONNX import TDDFA_ONNX
        from utils.tddfa_util import _parse_param

        with open(THREEDDFA_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tddfa = TDDFA_ONNX(**cfg)
    finally:
        os.chdir(old_cwd)
    return tddfa, _parse_param


def select_candidate_frames(stream: np.lib.npyio.NpzFile) -> np.ndarray:
    src = np.asarray(stream["mesh_source"]).astype(str)
    mask = (
        np.asarray(stream["geometry_observed"], dtype=bool)
        & (np.asarray(stream["mesh_conf"], dtype=np.float32) >= MIN_MESH_CONF)
        & (np.abs(np.asarray(stream["yaw_deg"], dtype=np.float32)) < YAW_GATE_DEG)
        & (np.asarray(stream["head_scale_px"], dtype=np.float32) >= MIN_HEAD_SCALE_PX)
        & np.char.startswith(src, "observed")
    )
    idx = np.where(mask)[0]
    selected: List[int] = []
    last = -10_000
    for i in idx:
        if int(i) - last >= MIN_FRAME_GAP:
            selected.append(int(i))
            last = int(i)
        if len(selected) >= MAX_SOLVE_FRAMES:
            break
    return np.asarray(selected, dtype=np.int32)


def bbox_from_projected(points_px: np.ndarray, fw: int, fh: int) -> Optional[np.ndarray]:
    pts = np.asarray(points_px, dtype=np.float32)
    good = np.isfinite(pts).all(axis=1)
    if int(good.sum()) < 100:
        return None
    p = pts[good]
    x1, y1 = np.min(p, axis=0)
    x2, y2 = np.max(p, axis=0)
    if x2 - x1 < 8.0 or y2 - y1 < 8.0:
        return None
    return np.asarray([
        np.clip(x1, 0.0, fw - 1.0),
        np.clip(y1, 0.0, fh - 1.0),
        np.clip(x2, 0.0, fw - 1.0),
        np.clip(y2, 0.0, fh - 1.0),
    ], dtype=np.float32)


def robust_weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out = np.zeros(vals.shape[1], dtype=np.float64)
    for j in range(vals.shape[1]):
        order = np.argsort(vals[:, j])
        xs = vals[order, j]
        ws = w[order]
        c = np.cumsum(ws)
        out[j] = xs[np.searchsorted(c, 0.5 * c[-1])]
    return out


def huber_coeff_aggregate(values: np.ndarray, base_weights: np.ndarray,
                          iters: int = 6, c: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64)
    w0 = np.asarray(base_weights, dtype=np.float64)
    alpha = robust_weighted_median(vals, w0)
    final_weights = w0.copy()
    for _ in range(iters):
        scale = np.median(np.abs(vals - alpha[None, :]), axis=0) * 1.4826 + 1e-6
        dist = np.linalg.norm((vals - alpha[None, :]) / scale[None, :], axis=1) / np.sqrt(vals.shape[1])
        robust = np.minimum(1.0, c / np.maximum(dist, 1e-9))
        final_weights = w0 * robust
        alpha = np.sum(vals * final_weights[:, None], axis=0) / max(float(np.sum(final_weights)), 1e-9)
    return alpha.astype(np.float32), final_weights.astype(np.float32)


def initial_residual_gate(rows: List[FitRow]) -> np.ndarray:
    residual = np.asarray([r.residual_over_scale for r in rows], dtype=np.float64)
    p90 = np.asarray([r.p90_over_scale for r in rows], dtype=np.float64)
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)) * 1.4826)
    return (
        (residual <= min(MAX_RESIDUAL_OVER_SCALE, med + 2.5 * max(mad, 1e-6)))
        & (p90 <= MAX_P90_OVER_SCALE)
    )


def coeff_distance_gate(alpha_rows: np.ndarray, alpha_center: np.ndarray) -> np.ndarray:
    scale = np.median(np.abs(alpha_rows - alpha_center[None, :]), axis=0) * 1.4826 + 1e-6
    dist = np.linalg.norm((alpha_rows - alpha_center[None, :]) / scale[None, :], axis=1) / np.sqrt(alpha_rows.shape[1])
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)) * 1.4826)
    return dist <= max(2.75, med + 2.5 * max(mad, 1e-6))


def bfm_vertices_from_alpha(alpha_shp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(BFM_PATH, "rb") as f:
        bfm = pickle.load(f)
    u = bfm["u"].astype(np.float32).reshape(-1, 3)
    w_shp = bfm["w_shp"].astype(np.float32)[:, :40].reshape(u.shape[0], 3, 40)
    verts = u + np.einsum("nck,k->nc", w_shp, np.asarray(alpha_shp, dtype=np.float32))
    return verts.astype(np.float32), u, w_shp


def fixed_pose_project(row: FitRow, alpha_shp: np.ndarray, tddfa, parse_param_fn) -> np.ndarray:
    from utils.tddfa_util import similar_transform

    R, offset, _old_shp, alpha_exp = parse_param_fn(row.param)
    pts3d = R @ (
        tddfa.u_base
        + tddfa.w_shp_base @ np.asarray(alpha_shp, dtype=np.float32).reshape(40, 1)
        + tddfa.w_exp_base @ alpha_exp
    ).reshape(3, -1, order="F") + offset
    pts3d = similar_transform(pts3d, row.roi_box.tolist(), tddfa.size)
    return pts3d.T[:, :2].astype(np.float32)


def reprojection_summary(rows: List[FitRow], stream: np.lib.npyio.NpzFile,
                         alpha_shp: np.ndarray, tddfa, parse_param_fn) -> Dict:
    if not rows:
        return {"n": 0}
    personalized = []
    generic = []
    for row in rows:
        target = np.asarray(stream["projected_px"][row.frame], dtype=np.float32)[DDFA68_TO_MP468]
        pred_personal = fixed_pose_project(row, alpha_shp, tddfa, parse_param_fn)
        pred_generic = fixed_pose_project(row, np.zeros(40, dtype=np.float32), tddfa, parse_param_fn)
        denom = max(float(row.head_scale_px), 1.0)
        personalized.append(float(np.mean(np.linalg.norm(pred_personal - target, axis=1)) / denom))
        generic.append(float(np.mean(np.linalg.norm(pred_generic - target, axis=1)) / denom))
    p = np.asarray(personalized, dtype=np.float64)
    g = np.asarray(generic, dtype=np.float64)
    return {
        "n": int(len(rows)),
        "personalized_mean": float(np.mean(p)),
        "personalized_median": float(np.median(p)),
        "generic_mean": float(np.mean(g)),
        "generic_median": float(np.median(g)),
        "personalized_beats_generic_mean": bool(float(np.mean(p)) < float(np.mean(g))),
        "mean_delta_personalized_minus_generic": float(np.mean(p - g)),
    }


def shape_stability_summary(alpha_rows: np.ndarray, alpha_shared: np.ndarray) -> Dict:
    shared, _u, w_shp = bfm_vertices_from_alpha(alpha_shared)
    head_scale = max(float(np.ptp(shared[:, 0])), float(np.ptp(shared[:, 1])), 1e-6)
    vals = []
    for alpha in alpha_rows:
        verts = shared + np.einsum("nck,k->nc", w_shp, np.asarray(alpha - alpha_shared, dtype=np.float32))
        dev = np.linalg.norm(verts - shared, axis=1)
        vals.append(float(np.median(dev) / head_scale))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "method": "median_per_vertex_identity_deviation_over_model_head_xy_span",
        "n": int(len(vals)),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "p90": float(np.percentile(arr, 90.0)) if len(arr) else float("nan"),
        "max": float(np.max(arr)) if len(arr) else float("nan"),
        "target_le_0p25": bool(len(arr) > 0 and float(np.percentile(arr, 90.0)) <= 0.25),
    }


def run() -> Dict:
    t0 = time.time()
    print(f"{LOG_PREFIX} loading stream")
    stream = np.load(STREAM_PATH, allow_pickle=True)
    pipeline_version = str(stream["pipeline_version"][0])
    if pipeline_version != "mesh_cascade_v4":
        raise RuntimeError(f"expected mesh_cascade_v4, got {pipeline_version}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {VIDEO_PATH}")
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{LOG_PREFIX} loading 3DDFA ONNX")
    tddfa, parse_param_fn = load_tddfa()
    candidate_frames = select_candidate_frames(stream)
    print(f"{LOG_PREFIX} candidate frames: {len(candidate_frames)}")

    rows: List[FitRow] = []
    rejected: List[Dict] = []
    for n, fidx in enumerate(candidate_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame_bgr = cap.read()
        if not ok:
            rejected.append({"frame": int(fidx), "reason": "video_read_failed"})
            continue
        box = bbox_from_projected(stream["projected_px"][fidx], fw, fh)
        if box is None:
            rejected.append({"frame": int(fidx), "reason": "bad_cascade_bbox"})
            continue
        try:
            param_lst, roi_box_lst = tddfa(frame_bgr, [box.tolist()])
            landmarks68 = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0].T.astype(np.float32)
            R, offset, alpha_shp, alpha_exp = parse_param_fn(param_lst[0])
            del R, offset
        except Exception as exc:
            rejected.append({"frame": int(fidx), "reason": f"3ddfa_error_{type(exc).__name__}"})
            continue
        target = np.asarray(stream["projected_px"][fidx], dtype=np.float32)[DDFA68_TO_MP468]
        err = np.linalg.norm(landmarks68[:, :2] - target, axis=1)
        denom = max(float(stream["head_scale_px"][fidx]), 1.0)
        rows.append(FitRow(
            frame=int(fidx),
            yaw_deg=float(stream["yaw_deg"][fidx]),
            head_scale_px=float(stream["head_scale_px"][fidx]),
            mesh_conf=float(stream["mesh_conf"][fidx]),
            mesh_source=str(stream["mesh_source"][fidx]),
            box=box.astype(np.float32),
            roi_box=np.asarray(roi_box_lst[0], dtype=np.float32),
            residual_over_scale=float(np.mean(err) / denom),
            p90_over_scale=float(np.percentile(err, 90.0) / denom),
            alpha_shp=alpha_shp.reshape(-1).astype(np.float32),
            alpha_exp=alpha_exp.reshape(-1).astype(np.float32),
            param=np.asarray(param_lst[0], dtype=np.float32),
        ))
        if n % 25 == 0:
            print(f"{LOG_PREFIX} fit {n:03d}/{len(candidate_frames)} frame={int(fidx)}")
    cap.release()

    if len(rows) < 20:
        raise RuntimeError(f"not enough successful 3DDFA rows: {len(rows)}")

    residual_mask = initial_residual_gate(rows)
    gated_rows = [r for r, keep in zip(rows, residual_mask) if bool(keep)]
    alpha_gated = np.stack([r.alpha_shp for r in gated_rows], axis=0)
    residual_gated = np.asarray([r.residual_over_scale for r in gated_rows], dtype=np.float64)
    base_weights = 1.0 / np.maximum(residual_gated, 0.03) ** 2

    alpha_first, weights_first = huber_coeff_aggregate(alpha_gated, base_weights)
    coeff_mask = coeff_distance_gate(alpha_gated, alpha_first)
    accepted_rows = [r for r, keep in zip(gated_rows, coeff_mask) if bool(keep)]
    alpha_accepted = np.stack([r.alpha_shp for r in accepted_rows], axis=0)
    residual_accepted = np.asarray([r.residual_over_scale for r in accepted_rows], dtype=np.float64)
    weights_accepted = 1.0 / np.maximum(residual_accepted, 0.03) ** 2
    alpha_shared, final_weights = huber_coeff_aggregate(alpha_accepted, weights_accepted)
    alpha_wmedian = robust_weighted_median(alpha_accepted, weights_accepted).astype(np.float32)

    train_rows = [r for i, r in enumerate(accepted_rows) if i % 5 != 0]
    heldout_rows = [r for i, r in enumerate(accepted_rows) if i % 5 == 0]
    if len(train_rows) >= 20 and len(heldout_rows) >= 4:
        alpha_train, _train_w = huber_coeff_aggregate(
            np.stack([r.alpha_shp for r in train_rows], axis=0),
            1.0 / np.maximum([r.residual_over_scale for r in train_rows], 0.03) ** 2,
        )
    else:
        alpha_train = alpha_shared
        heldout_rows = accepted_rows

    stability = shape_stability_summary(alpha_accepted, alpha_shared)
    train_reproj = reprojection_summary(train_rows, stream, alpha_train, tddfa, parse_param_fn)
    heldout_reproj = reprojection_summary(heldout_rows, stream, alpha_train, tddfa, parse_param_fn)

    with open(TRI_PATH, "rb") as f:
        tri_raw = pickle.load(f)
    tri = np.asarray(tri_raw.T if tri_raw.shape[0] == 3 else tri_raw, dtype=np.int32)
    verts, _u, _w = bfm_vertices_from_alpha(alpha_shared)

    all_rows_alpha = np.stack([r.alpha_shp for r in rows], axis=0)
    accepted_frame_set = {r.frame for r in accepted_rows}
    train_frame_set = {r.frame for r in train_rows}
    heldout_frame_set = {r.frame for r in heldout_rows}
    accepted_mask_all = np.asarray([r.frame in accepted_frame_set for r in rows], dtype=bool)
    train_mask_all = np.asarray([r.frame in train_frame_set for r in rows], dtype=bool)
    heldout_mask_all = np.asarray([r.frame in heldout_frame_set for r in rows], dtype=bool)

    np.savez_compressed(
        IDENTITY_PATH,
        pipeline_version=np.asarray([PIPELINE_VERSION]),
        generated_at_unix=np.asarray([time.time()], dtype=np.float64),
        alpha_shp=alpha_shared.astype(np.float32),
        alpha_shp_train=alpha_train.astype(np.float32),
        alpha_shp_weighted_median=alpha_wmedian.astype(np.float32),
        accepted_frames=np.asarray([r.frame for r in accepted_rows], dtype=np.int32),
        train_frames=np.asarray([r.frame for r in train_rows], dtype=np.int32),
        heldout_frames=np.asarray([r.frame for r in heldout_rows], dtype=np.int32),
        candidate_frames=np.asarray([r.frame for r in rows], dtype=np.int32),
        candidate_alpha_shp=all_rows_alpha.astype(np.float32),
        candidate_alpha_exp=np.stack([r.alpha_exp for r in rows], axis=0).astype(np.float32),
        candidate_param=np.stack([r.param for r in rows], axis=0).astype(np.float32),
        candidate_roi_box=np.stack([r.roi_box for r in rows], axis=0).astype(np.float32),
        candidate_box=np.stack([r.box for r in rows], axis=0).astype(np.float32),
        candidate_residual_over_scale=np.asarray([r.residual_over_scale for r in rows], dtype=np.float32),
        candidate_p90_over_scale=np.asarray([r.p90_over_scale for r in rows], dtype=np.float32),
        accepted_mask=accepted_mask_all,
        train_mask=train_mask_all,
        heldout_mask=heldout_mask_all,
        accepted_weights=final_weights.astype(np.float32),
        bfm_vertex_count=np.asarray([verts.shape[0]], dtype=np.int32),
        bfm_tri_count=np.asarray([tri.shape[0]], dtype=np.int32),
        bfm_license=np.asarray(["academic-use-only: _deps/3DDFA_V2/bfm/readme.md"]),
        source_video=np.asarray([VIDEO_PATH]),
        stream_path=np.asarray([STREAM_PATH]),
        threeddfa_repo=np.asarray([THREEDDFA_REPO]),
    )

    residual_all = np.asarray([r.residual_over_scale for r in rows], dtype=np.float64)
    residual_acc = np.asarray([r.residual_over_scale for r in accepted_rows], dtype=np.float64)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_unix": time.time(),
        "paths": {
            "identity_npz": IDENTITY_PATH,
            "report": REPORT_PATH,
            "video": VIDEO_PATH,
            "stream": STREAM_PATH,
            "threeddfa_repo": THREEDDFA_REPO,
            "bfm": BFM_PATH,
        },
        "frame_selection": {
            "source": "mesh_cascade_v4 observed cascade landmarks; no FaceBoxes",
            "candidate_count": int(len(candidate_frames)),
            "successful_3ddfa_rows": int(len(rows)),
            "residual_gated_count": int(len(gated_rows)),
            "accepted_count": int(len(accepted_rows)),
            "train_count": int(len(train_rows)),
            "heldout_count": int(len(heldout_rows)),
            "gates": {
                "abs_yaw_lt_deg": YAW_GATE_DEG,
                "mesh_conf_gte": MIN_MESH_CONF,
                "head_scale_px_gte": MIN_HEAD_SCALE_PX,
                "residual_over_scale_lte": MAX_RESIDUAL_OVER_SCALE,
                "p90_over_scale_lte": MAX_P90_OVER_SCALE,
            },
        },
        "fit_residual_over_head_scale": {
            "all_median": float(np.median(residual_all)),
            "all_p90": float(np.percentile(residual_all, 90.0)),
            "accepted_median": float(np.median(residual_acc)),
            "accepted_p90": float(np.percentile(residual_acc, 90.0)),
        },
        "identity_aggregation": {
            "method": "residual gate + robust coefficient gate + weighted-median initialized Huber aggregate",
            "alpha_shp_norm": float(np.linalg.norm(alpha_shared)),
            "weighted_median_norm": float(np.linalg.norm(alpha_wmedian)),
            "per_frame_alpha_norm_median": float(np.median(np.linalg.norm(alpha_accepted, axis=1))),
            "per_frame_alpha_norm_p90": float(np.percentile(np.linalg.norm(alpha_accepted, axis=1), 90.0)),
        },
        "identity_stability": stability,
        "fixed_pose_reprojection": {
            "note": "Uses fixed 3DDFA pose/expression and swaps only alpha_shp; sparse 2D landmarks are weakly sensitive to BFM identity.",
            "train": train_reproj,
            "heldout": heldout_reproj,
        },
        "license_flag": {
            "bfm_asset": "_deps/3DDFA_V2/configs/bfm_noneck_v3.pkl",
            "status": "academic-use-only",
            "source": "_deps/3DDFA_V2/bfm/readme.md",
            "commercial_clean": False,
        },
        "mps_renderer_dependency_boundary": {
            "uses_3ddfa_onnxruntime_cpu": True,
            "uses_faceboxes": False,
            "uses_flame_or_mica": False,
            "uses_disallowed_gpu_rasterizer": False,
        },
        "rejected_rows_sample": rejected[:20],
        "processing_time_s": round(time.time() - t0, 3),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"{LOG_PREFIX} saved {IDENTITY_PATH}")
    print(f"{LOG_PREFIX} accepted={len(accepted_rows)} stability_p90={stability['p90']:.4f}")
    print(f"{LOG_PREFIX} heldout personalized_mean={heldout_reproj.get('personalized_mean', float('nan')):.4f} generic_mean={heldout_reproj.get('generic_mean', float('nan')):.4f}")
    return report


if __name__ == "__main__":
    run()
