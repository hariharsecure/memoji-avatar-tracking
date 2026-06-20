#!/usr/bin/env python3
"""
bfm_avatar_overlay_v1.py

Dense personalized BFM avatar overlay for the subject likeness loop.

The renderer reuses the existing v16 screen-space transform and alpha
compositing conventions, but replaces the generic emoji GLB with the 3DDFA_V2
no-neck BFM mesh. The identity shape is fixed for every frame.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pyrender
import trimesh

from avatar_overlay_pipeline_v16 import (
    FOCAL_LEN,
    build_T_from_screen_pos,
    composite_rgba_bgr,
)
from identity_bfm_solve_v1 import (
    BFM_PATH,
    DDFA68_TO_MP468,
    IDENTITY_PATH,
    OUT_DIR,
    STREAM_PATH,
    THREEDDFA_REPO,
    TRI_PATH,
    VIDEO_PATH,
    bbox_from_projected,
    load_tddfa,
)


CONTROL_PATH = f"{OUT_DIR}/mesh_avatar_driver_v1_controls.npz"
OLD_GENERIC_PROOF_PREFIX = f"{OUT_DIR}/mesh_avatar_driver_v1_proof"

RAW_PATH = f"{OUT_DIR}/subject_bfm_avatar_v1_raw.mp4"
MASTER_PATH = f"{OUT_DIR}/subject_bfm_avatar_v1_3up_master.mp4"
PREVIEW_PATH = f"{OUT_DIR}/subject_bfm_avatar_v1_3up_preview.mp4"
CONTACT_SHEET_PATH = f"{OUT_DIR}/subject_bfm_likeness_contact_sheet_v1.png"
REPORT_PATH = f"{OUT_DIR}/subject_bfm_avatar_v1_report.json"
NOTES_PATH = f"{OUT_DIR}/notes_bfm_avatar_v1.md"
EXPR_CACHE_PATH = f"{OUT_DIR}/subject_bfm_expression_track_v1.npz"

PIPELINE_VERSION = "bfm_avatar_overlay_v1"
LOG_PREFIX = "[bfm-avatar-v1]"

PANEL_W = 360
PANEL_H = 640
Z_REF = -88.8
BFM_SCALE_K = 0.0640
BFM_SCALE_MIN = 3.0
BFM_SCALE_MAX = 25.0

EXPR_RELIABLE_YAW_DEG = 55.0
EXPR_MIN_MESH_CONF = 0.85
EXPR_MIN_HEAD_SCALE = 30.0
EXPR_EMA_OBSERVED = 0.55
EXPR_EMA_PROFILE = 0.28
EXPR_EMA_INTERP = 0.14

SOURCE_TINT_RGB = {
    "observed": None,
    "profile": np.asarray([1.0, 0.62, 0.08], dtype=np.float32),
    "interpolated": np.asarray([0.10, 0.45, 1.0], dtype=np.float32),
}

CONTACT_FRAMES = [
    (50, "frontal"),
    (89, "three_quarter"),
    (188, "profile"),
    (828, "close_up"),
]

OLD_GENERIC_PROOFS = {
    50: f"{OLD_GENERIC_PROOF_PREFIX}_frontal_f0050.jpg",
    89: f"{OLD_GENERIC_PROOF_PREFIX}_three_quarter_f0089.jpg",
    188: f"{OLD_GENERIC_PROOF_PREFIX}_profile_f0188.jpg",
    828: f"{OLD_GENERIC_PROOF_PREFIX}_close_up_f0828.jpg",
}


@dataclass
class Stream:
    frame: np.ndarray
    projected_px: np.ndarray
    mesh_source: np.ndarray
    mesh_conf: np.ndarray
    geometry_observed: np.ndarray
    yaw_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    head_transform: np.ndarray
    head_center_px: np.ndarray
    head_scale_px: np.ndarray


@dataclass
class Controls:
    frame: np.ndarray
    controls: np.ndarray
    target_names: List[str]
    center_px: np.ndarray
    head_scale_px: np.ndarray
    yaw_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    mesh_source: np.ndarray


@dataclass
class BFMGeometry:
    u: np.ndarray
    w_shp: np.ndarray
    w_exp: np.ndarray
    faces: np.ndarray
    dropped_face_count: int
    keypoint_vertices: np.ndarray
    identity_base_local: np.ndarray
    generic_base_local: np.ndarray
    w_exp_local: np.ndarray
    center_model: np.ndarray
    scale_model: float
    base_normals: np.ndarray


@dataclass
class ColorModel:
    skin_rgb: np.ndarray
    brow_rgb: np.ndarray
    hair_rgb: np.ndarray
    facial_rgb: np.ndarray
    lip_rgb: np.ndarray
    feature_enabled: Dict[str, bool]
    sample_counts: Dict[str, int]
    sample_std: Dict[str, float]
    stylized_vertex_rgba: np.ndarray
    generic_vertex_rgba: np.ndarray
    mask_counts: Dict[str, int]


def source_class(source: str) -> str:
    s = str(source)
    if s == "interpolated":
        return "interpolated"
    if s == "profile_fit":
        return "profile"
    return "observed"


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_avatar_scale(head_scale_px: float) -> float:
    return float(np.clip(BFM_SCALE_K * max(float(head_scale_px), 10.0), BFM_SCALE_MIN, BFM_SCALE_MAX))


def load_stream() -> Stream:
    z = np.load(STREAM_PATH, allow_pickle=True)
    if str(z["pipeline_version"][0]) != "mesh_cascade_v4":
        raise RuntimeError("Expected mesh_cascade_v4 stream")
    return Stream(
        frame=np.asarray(z["frame"], dtype=np.int32),
        projected_px=np.asarray(z["projected_px"], dtype=np.float32),
        mesh_source=np.asarray(z["mesh_source"]).astype(str),
        mesh_conf=np.asarray(z["mesh_conf"], dtype=np.float32),
        geometry_observed=np.asarray(z["geometry_observed"], dtype=bool),
        yaw_deg=np.asarray(z["yaw_deg"], dtype=np.float32),
        pitch_deg=np.asarray(z["pitch_deg"], dtype=np.float32),
        roll_deg=np.asarray(z["roll_deg"], dtype=np.float32),
        head_transform=np.asarray(z["head_transform"], dtype=np.float32),
        head_center_px=np.asarray(z["head_center_px"], dtype=np.float32),
        head_scale_px=np.asarray(z["head_scale_px"], dtype=np.float32),
    )


def load_controls() -> Controls:
    z = np.load(CONTROL_PATH, allow_pickle=True)
    return Controls(
        frame=np.asarray(z["frame"], dtype=np.int32),
        controls=np.asarray(z["controls"], dtype=np.float32),
        target_names=[str(x) for x in z["target_names"]],
        center_px=np.asarray(z["center_px"], dtype=np.float32),
        head_scale_px=np.asarray(z["head_scale_px"], dtype=np.float32),
        yaw_deg=np.asarray(z["yaw_deg"], dtype=np.float32),
        pitch_deg=np.asarray(z["pitch_deg"], dtype=np.float32),
        roll_deg=np.asarray(z["roll_deg"], dtype=np.float32),
        mesh_source=np.asarray(z["mesh_source"]).astype(str),
    )


def unit_rotation_from_transform(T: np.ndarray) -> np.ndarray:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    s = max(float(np.linalg.norm(R[:, 0])), 1e-8)
    R = R / s
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1.0
    return R


def load_bfm_geometry(alpha_shp: np.ndarray) -> BFMGeometry:
    with open(BFM_PATH, "rb") as f:
        bfm = pickle.load(f)
    u = bfm["u"].astype(np.float32).reshape(-1, 3)
    n = u.shape[0]
    w_shp = bfm["w_shp"].astype(np.float32)[:, :40].reshape(n, 3, 40)
    w_exp = bfm["w_exp"].astype(np.float32)[:, :10].reshape(n, 3, 10)

    with open(TRI_PATH, "rb") as f:
        tri_raw = pickle.load(f)
    faces = np.asarray(tri_raw.T if tri_raw.shape[0] == 3 else tri_raw, dtype=np.int64)
    if faces.size and faces.min() >= 1 and faces.max() <= n:
        faces = faces - 1
    good = np.all((faces >= 0) & (faces < n), axis=1)
    dropped = int((~good).sum())
    faces = faces[good].astype(np.int32)

    identity_model = u + np.einsum("nck,k->nc", w_shp, np.asarray(alpha_shp, dtype=np.float32))
    generic_model = u.copy()
    mn = identity_model.min(axis=0)
    mx = identity_model.max(axis=0)
    center = 0.5 * (mn + mx)
    scale = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]), 1e-6) * 0.5
    identity_local = (identity_model - center[None, :]) / scale
    generic_local = (generic_model - center[None, :]) / scale
    w_exp_local = w_exp / scale

    key = bfm["keypoints"].astype(np.int64).reshape(68, 3)
    key_vertices = (key[:, 0] // 3).astype(np.int32)

    mesh = trimesh.Trimesh(vertices=identity_local, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    return BFMGeometry(
        u=u,
        w_shp=w_shp,
        w_exp=w_exp,
        faces=faces,
        dropped_face_count=dropped,
        keypoint_vertices=key_vertices,
        identity_base_local=identity_local.astype(np.float32),
        generic_base_local=generic_local.astype(np.float32),
        w_exp_local=w_exp_local.astype(np.float32),
        center_model=center.astype(np.float32),
        scale_model=float(scale),
        base_normals=normals,
    )


def vertices_with_expression(base_local: np.ndarray, w_exp_local: np.ndarray,
                             alpha_exp: np.ndarray) -> np.ndarray:
    return (
        base_local
        + np.einsum("nck,k->nc", w_exp_local, np.asarray(alpha_exp, dtype=np.float32))
    ).astype(np.float32)


def sample_patch_rgb(frame_bgr: np.ndarray, xy: np.ndarray, radius: int = 3) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    rgb = patch.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0
    return np.median(rgb, axis=0)


def color_stats(samples: List[np.ndarray], fallback: Iterable[float]) -> Tuple[np.ndarray, int, float]:
    if not samples:
        return np.asarray(fallback, dtype=np.float32), 0, float("inf")
    arr = np.asarray(samples, dtype=np.float32)
    med = np.median(arr, axis=0)
    std = float(np.median(np.std(arr, axis=0)))
    return med.astype(np.float32), int(arr.shape[0]), std


def sample_video_colors(stream: Stream) -> Dict:
    skin_idx = np.asarray([50, 101, 118, 123, 147, 187, 205, 425, 352, 376, 330, 280], dtype=np.int32)
    brow_idx = np.asarray([70, 63, 105, 66, 107, 336, 296, 334, 293, 300], dtype=np.int32)
    hair_idx = np.asarray([10, 67, 109, 297, 338], dtype=np.int32)
    facial_idx = np.asarray([0, 17, 57, 287, 164, 200, 152], dtype=np.int32)

    frontal = np.where(
        stream.geometry_observed
        & (stream.mesh_conf >= 0.95)
        & (np.abs(stream.yaw_deg) < 12.0)
        & (stream.head_scale_px >= 60.0)
    )[0]
    if len(frontal) > 48:
        frontal = frontal[np.linspace(0, len(frontal) - 1, 48).round().astype(np.int32)]

    cap = cv2.VideoCapture(VIDEO_PATH)
    groups = {"skin": [], "brow": [], "hair": [], "facial": []}
    for fidx in frontal:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok:
            continue
        pts = stream.projected_px[fidx]
        for name, idxs in [("skin", skin_idx), ("brow", brow_idx), ("hair", hair_idx), ("facial", facial_idx)]:
            for idx in idxs:
                rgb = sample_patch_rgb(frame, pts[int(idx)])
                if rgb is not None and np.isfinite(rgb).all():
                    groups[name].append(rgb)
    cap.release()

    skin, skin_n, skin_std = color_stats(groups["skin"], [0.64, 0.42, 0.30])
    brow, brow_n, brow_std = color_stats(groups["brow"], [0.06, 0.045, 0.035])
    hair, hair_n, hair_std = color_stats(groups["hair"], [0.055, 0.04, 0.03])
    facial, facial_n, facial_std = color_stats(groups["facial"], [0.075, 0.055, 0.045])

    # Keep the primary avatar stylized rather than photoreal: lift skin gently
    # but preserve the sampled hue.
    skin = np.clip(0.68 * skin + 0.32 * np.asarray([0.72, 0.50, 0.38], dtype=np.float32), 0.0, 1.0)
    dark_default = np.asarray([0.045, 0.032, 0.024], dtype=np.float32)
    brow_enabled = brow_n >= 20 and brow_std <= 0.28
    hair_enabled = hair_n >= 12 and hair_std <= 0.24
    facial_enabled = facial_n >= 20 and facial_std <= 0.22
    if not brow_enabled or float(np.dot(brow, [0.299, 0.587, 0.114])) > 0.32:
        brow = dark_default
    if not hair_enabled or float(np.dot(hair, [0.299, 0.587, 0.114])) > 0.26:
        hair = dark_default
    if not facial_enabled or float(np.dot(facial, [0.299, 0.587, 0.114])) > 0.20:
        facial = dark_default
    lip = np.clip(0.72 * skin + 0.28 * np.asarray([0.52, 0.11, 0.12], dtype=np.float32), 0.0, 1.0)

    return {
        "skin": skin,
        "brow": brow.astype(np.float32),
        "hair": hair.astype(np.float32),
        "facial": facial.astype(np.float32),
        "lip": lip.astype(np.float32),
        "feature_enabled": {
            "brow": bool(brow_enabled),
            "hair": bool(hair_enabled),
            "facial_hair": bool(facial_enabled),
        },
        "sample_counts": {k: int(len(v)) for k, v in groups.items()},
        "sample_std": {
            "skin": skin_std,
            "brow": brow_std,
            "hair": hair_std,
            "facial": facial_std,
        },
    }


def ellipse_mask(local: np.ndarray, center: np.ndarray, rx: float, ry: float,
                 front: Optional[np.ndarray] = None) -> np.ndarray:
    dx = (local[:, 0] - float(center[0])) / max(float(rx), 1e-6)
    dy = (local[:, 1] - float(center[1])) / max(float(ry), 1e-6)
    mask = (dx * dx + dy * dy) <= 1.0
    if front is not None:
        mask &= front
    return mask


def make_vertex_colors(geom: BFMGeometry, sampled: Dict, base_local: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    local = np.asarray(base_local, dtype=np.float32)
    k = geom.keypoint_vertices
    lm = local[k]
    skin = sampled["skin"]
    brow = sampled["brow"]
    hair = sampled["hair"]
    facial = sampled["facial"]
    lip = sampled["lip"]
    eye = np.asarray([0.035, 0.026, 0.022], dtype=np.float32)

    colors = np.tile(np.r_[skin, 1.0].astype(np.float32), (local.shape[0], 1))
    front = local[:, 2] >= np.quantile(local[:, 2], 0.54)

    left_brow_c = np.mean(lm[17:22], axis=0)
    right_brow_c = np.mean(lm[22:27], axis=0)
    left_eye_c = np.mean(lm[36:42], axis=0)
    right_eye_c = np.mean(lm[42:48], axis=0)
    mouth_c = np.mean(lm[48:68], axis=0)
    mouth_w = max(float(np.linalg.norm(lm[54] - lm[48])), 0.18)
    brow_w_l = max(float(np.linalg.norm(lm[21] - lm[17])), 0.18)
    brow_w_r = max(float(np.linalg.norm(lm[26] - lm[22])), 0.18)
    eye_w_l = max(float(np.linalg.norm(lm[39] - lm[36])), 0.12)
    eye_w_r = max(float(np.linalg.norm(lm[45] - lm[42])), 0.12)

    hair_line_y = max(float(left_brow_c[1]), float(right_brow_c[1])) + 0.16
    hair_mask = local[:, 1] >= hair_line_y
    brow_mask = (
        ellipse_mask(local, left_brow_c + np.asarray([0.0, 0.045, 0.0], dtype=np.float32), brow_w_l * 0.62, 0.055, front)
        | ellipse_mask(local, right_brow_c + np.asarray([0.0, 0.045, 0.0], dtype=np.float32), brow_w_r * 0.62, 0.055, front)
    )
    eye_mask = (
        ellipse_mask(local, left_eye_c, eye_w_l * 0.45, 0.055, front)
        | ellipse_mask(local, right_eye_c, eye_w_r * 0.45, 0.055, front)
    )
    mouth_mask = ellipse_mask(local, mouth_c, mouth_w * 0.48, 0.075, front)
    moustache_c = mouth_c + np.asarray([0.0, 0.105, 0.0], dtype=np.float32)
    moustache_mask = ellipse_mask(local, moustache_c, mouth_w * 0.43, 0.060, front)
    chin_y = float(lm[8, 1])
    beard_mask = (
        (local[:, 1] <= mouth_c[1] - 0.055)
        & (local[:, 1] >= chin_y - 0.06)
        & (np.abs(local[:, 0]) <= mouth_w * 1.28)
        & front
    )

    if sampled["feature_enabled"].get("hair", False):
        colors[hair_mask, :3] = hair
    if sampled["feature_enabled"].get("facial_hair", False):
        colors[beard_mask, :3] = 0.35 * skin + 0.65 * facial
        colors[moustache_mask, :3] = facial
    if sampled["feature_enabled"].get("brow", False):
        colors[brow_mask, :3] = brow
    colors[eye_mask, :3] = eye
    colors[mouth_mask, :3] = lip

    counts = {
        "hair": int(hair_mask.sum()),
        "brow": int(brow_mask.sum()),
        "eye": int(eye_mask.sum()),
        "mouth": int(mouth_mask.sum()),
        "moustache": int(moustache_mask.sum()),
        "beard": int(beard_mask.sum()),
    }
    return np.clip(colors * 255.0, 0, 255).astype(np.uint8), counts


def build_color_model(stream: Stream, geom: BFMGeometry) -> ColorModel:
    sampled = sample_video_colors(stream)
    stylized, counts = make_vertex_colors(geom, sampled, geom.identity_base_local)
    generic, _counts_generic = make_vertex_colors(geom, sampled, geom.generic_base_local)
    return ColorModel(
        skin_rgb=sampled["skin"],
        brow_rgb=sampled["brow"],
        hair_rgb=sampled["hair"],
        facial_rgb=sampled["facial"],
        lip_rgb=sampled["lip"],
        feature_enabled=sampled["feature_enabled"],
        sample_counts=sampled["sample_counts"],
        sample_std=sampled["sample_std"],
        stylized_vertex_rgba=stylized,
        generic_vertex_rgba=generic,
        mask_counts=counts,
    )


class BFMRenderer:
    def __init__(self, fw: int, fh: int, focal: float = FOCAL_LEN):
        self.fw = fw
        self.fh = fh
        self.focal = focal
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=[0.46, 0.46, 0.46],
        )
        cam = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=fw / 2.0, cy=fh / 2.0, znear=0.5, zfar=10000.0)
        self.scene.add(cam, pose=np.eye(4))
        for pos, col, intensity in [
            ([0, 0, 220], [1.0, 1.0, 1.0], 3.2),
            ([-120, 80, 180], [0.82, 0.90, 1.0], 1.8),
            ([100, -40, 120], [1.0, 0.92, 0.76], 1.0),
        ]:
            light = pyrender.DirectionalLight(color=col, intensity=intensity)
            pose = np.eye(4)
            pose[:3, 3] = pos
            self.scene.add(light, pose=pose)
        self.renderer = pyrender.OffscreenRenderer(fw, fh)

    def render(self, vertices: np.ndarray, faces: np.ndarray, T_norm: np.ndarray,
               scale: float, vertex_rgba: Optional[np.ndarray] = None,
               material_color: Optional[np.ndarray] = None) -> np.ndarray:
        S = np.eye(4, dtype=np.float64)
        S[0, 0] = S[1, 1] = S[2, 2] = float(scale)
        T_node = T_norm @ S
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        material = None
        if vertex_rgba is not None:
            mesh.visual.vertex_colors = vertex_rgba
        else:
            color = np.asarray(material_color if material_color is not None else [0.65, 0.48, 0.35, 1.0], dtype=np.float32)
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=color.tolist(),
                metallicFactor=0.0,
                roughnessFactor=0.92,
                smooth=False,
            )
        pm = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        node = self.scene.add(pm, pose=T_node)
        color, _depth = self.renderer.render(
            self.scene,
            flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
        self.scene.remove_node(node)
        return color

    def close(self) -> None:
        self.renderer.delete()


def tint_rgba_by_source(rgba: np.ndarray, source: str) -> np.ndarray:
    tint = SOURCE_TINT_RGB[source_class(source)]
    if tint is None:
        return rgba
    out = rgba.copy()
    mask = out[:, :, 3] > 0
    rgb = out[:, :, :3].astype(np.float32) / 255.0
    strength = 0.32 if source_class(source) == "profile" else 0.42
    rgb[mask] = (1.0 - strength) * rgb[mask] + strength * tint[None, :]
    out[:, :, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return out


def alpha_bbox(rgba: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_text(img: np.ndarray, text: str, xy: Tuple[int, int],
              color: Tuple[int, int, int] = (255, 255, 255),
              scale: float = 0.48, thickness: int = 1) -> None:
    x, y = xy
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def panelize(real: np.ndarray, stylized: np.ndarray, photo: np.ndarray) -> np.ndarray:
    cells = []
    for img in [real, stylized, photo]:
        cells.append(cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA))
    out = np.concatenate(cells, axis=1)
    draw_text(out, "real", (12, 28), (255, 255, 255), 0.58, 1)
    draw_text(out, "personalized stylized", (PANEL_W + 12, 28), (80, 255, 180), 0.52, 1)
    draw_text(out, "photo projection", (2 * PANEL_W + 12, 28), (80, 210, 255), 0.52, 1)
    return out


def video_meta() -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {VIDEO_PATH}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, fw, fh, n


def reliable_expr_mask(stream: Stream) -> np.ndarray:
    src = np.asarray(stream.mesh_source).astype(str)
    return (
        stream.geometry_observed
        & np.char.startswith(src, "observed")
        & (stream.mesh_conf >= EXPR_MIN_MESH_CONF)
        & (np.abs(stream.yaw_deg) <= EXPR_RELIABLE_YAW_DEG)
        & (stream.head_scale_px >= EXPR_MIN_HEAD_SCALE)
    )


def fit_expression_track(stream: Stream, controls: Controls, fw: int, fh: int) -> Dict:
    if os.path.exists(EXPR_CACHE_PATH):
        try:
            z = np.load(EXPR_CACHE_PATH, allow_pickle=True)
            if (
                str(z["pipeline_version"][0]) == PIPELINE_VERSION
                and int(z["frame_count"][0]) == len(stream.frame)
            ):
                return {
                    "alpha_exp": np.asarray(z["alpha_exp"], dtype=np.float32),
                    "raw_alpha_exp": np.asarray(z["raw_alpha_exp"], dtype=np.float32),
                    "valid_3ddfa": np.asarray(z["valid_3ddfa"], dtype=bool),
                    "expr_source": np.asarray(z["expr_source"]).astype(str),
                    "ridge_coef": np.asarray(z["ridge_coef"], dtype=np.float32),
                    "report": json.loads(str(z["report_json"][0])),
                }
        except Exception:
            pass

    print(f"{LOG_PREFIX} fitting 3DDFA expression track")
    tddfa, parse_param_fn = load_tddfa()
    mask = reliable_expr_mask(stream)
    raw = np.full((len(stream.frame), 10), np.nan, dtype=np.float32)
    valid = np.zeros(len(stream.frame), dtype=bool)

    cap = cv2.VideoCapture(VIDEO_PATH)
    for i in range(len(stream.frame)):
        if not bool(mask[i]):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        box = bbox_from_projected(stream.projected_px[i], fw, fh)
        if box is None:
            continue
        try:
            params, _rois = tddfa(frame, [box.tolist()])
            _R, _off, _shp, exp = parse_param_fn(params[0])
            raw[i] = exp.reshape(-1).astype(np.float32)
            valid[i] = True
        except Exception:
            continue
        if i % 100 == 0:
            print(f"{LOG_PREFIX} expression f{i:04d} valid={int(valid.sum())}")
    cap.release()

    if int(valid.sum()) < 20:
        raise RuntimeError(f"too few valid 3DDFA expression frames: {int(valid.sum())}")

    X = np.asarray(controls.controls, dtype=np.float64)
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    y = raw[valid].astype(np.float64)
    xv = X1[valid]
    lam = 1e-3 * float(np.trace(xv.T @ xv) / max(xv.shape[1], 1))
    coef = np.linalg.solve(xv.T @ xv + lam * np.eye(xv.shape[1]), xv.T @ y)
    pred = (X1 @ coef).astype(np.float32)

    lo = np.nanpercentile(raw[valid], 1.0, axis=0).astype(np.float32)
    hi = np.nanpercentile(raw[valid], 99.0, axis=0).astype(np.float32)
    span = np.maximum(hi - lo, 0.25)
    pred = np.clip(pred, lo - 0.25 * span, hi + 0.25 * span)

    target = np.zeros_like(pred, dtype=np.float32)
    expr_source = np.empty(len(stream.frame), dtype="<U24")
    for i in range(len(stream.frame)):
        cls = source_class(stream.mesh_source[i])
        if valid[i]:
            target[i] = raw[i]
            expr_source[i] = "observed_3ddfa"
        elif cls == "profile":
            target[i] = pred[i]
            expr_source[i] = "profile_driver_map"
        else:
            target[i] = pred[i] if i == 0 else target[i - 1] * 0.86 + pred[i] * 0.14
            expr_source[i] = "held_decay"

    smooth = np.zeros_like(target, dtype=np.float32)
    smooth[0] = target[0]
    for i in range(1, len(stream.frame)):
        cls = source_class(stream.mesh_source[i])
        if valid[i]:
            alpha = EXPR_EMA_OBSERVED
        elif cls == "profile":
            alpha = EXPR_EMA_PROFILE
        else:
            alpha = EXPR_EMA_INTERP
        smooth[i] = ((1.0 - alpha) * smooth[i - 1] + alpha * target[i]).astype(np.float32)

    report = {
        "valid_3ddfa_expr_frames": int(valid.sum()),
        "reliable_mask_frames": int(mask.sum()),
        "fallback_frames": int((~valid).sum()),
        "ridge_controls": controls.target_names,
        "ema": {
            "observed": EXPR_EMA_OBSERVED,
            "profile": EXPR_EMA_PROFILE,
            "interpolated": EXPR_EMA_INTERP,
        },
    }
    np.savez_compressed(
        EXPR_CACHE_PATH,
        pipeline_version=np.asarray([PIPELINE_VERSION]),
        frame_count=np.asarray([len(stream.frame)], dtype=np.int32),
        alpha_exp=smooth.astype(np.float32),
        raw_alpha_exp=raw.astype(np.float32),
        valid_3ddfa=valid,
        expr_source=expr_source,
        ridge_coef=coef.astype(np.float32),
        report_json=np.asarray([json.dumps(report)]),
    )
    return {
        "alpha_exp": smooth,
        "raw_alpha_exp": raw,
        "valid_3ddfa": valid,
        "expr_source": expr_source,
        "ridge_coef": coef.astype(np.float32),
        "report": report,
    }


def mouth_aperture_stream(points_px: np.ndarray) -> float:
    face_h = max(abs(float(points_px[152, 1]) - float(points_px[10, 1])), 1e-6)
    return abs(float(points_px[14, 1]) - float(points_px[13, 1])) / face_h


def mouth_aperture_bfm(geom: BFMGeometry, vertices_local: np.ndarray) -> float:
    k = geom.keypoint_vertices
    lm = vertices_local[k]
    face_h = max(abs(float(lm[27, 1]) - float(lm[8, 1])), 1e-6)
    return abs(float(lm[66, 1]) - float(lm[62, 1])) / face_h


def pearsonr_np(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if len(xa) < 3:
        return float("nan")
    xa = xa - np.mean(xa)
    ya = ya - np.mean(ya)
    denom = float(np.linalg.norm(xa) * np.linalg.norm(ya))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(xa, ya) / denom)


def summarize_array(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def project_vertices(vertices: np.ndarray, T_norm: np.ndarray, scale: float,
                     fw: int, fh: int, focal: float = FOCAL_LEN) -> Tuple[np.ndarray, np.ndarray]:
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = S[1, 1] = S[2, 2] = float(scale)
    T_node = T_norm @ S
    homo = np.concatenate([vertices.astype(np.float64), np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (T_node @ homo.T).T[:, :3]
    zneg = np.maximum(-cam[:, 2], 1e-6)
    px = np.empty((vertices.shape[0], 2), dtype=np.float32)
    px[:, 0] = fw / 2.0 + focal * cam[:, 0] / zneg
    px[:, 1] = fh / 2.0 - focal * cam[:, 1] / zneg
    return px, cam.astype(np.float32)


def build_T_for_frame(stream: Stream, controls: Controls, i: int, fw: int, fh: int) -> Tuple[np.ndarray, float, np.ndarray]:
    R = unit_rotation_from_transform(stream.head_transform[i])
    center = np.asarray(controls.center_px[i], dtype=np.float64)
    scale = compute_avatar_scale(float(controls.head_scale_px[i]))
    T_norm = build_T_from_screen_pos(R, float(center[0]), float(center[1]), fw, fh, FOCAL_LEN, Z_REF)
    return T_norm, scale, center


def choose_best_frontal(stream: Stream) -> int:
    score = (
        np.abs(stream.yaw_deg).astype(np.float64)
        - 0.002 * stream.head_scale_px.astype(np.float64)
        - 4.0 * stream.mesh_conf.astype(np.float64)
    )
    mask = stream.geometry_observed & (np.abs(stream.yaw_deg) < 12.0) & (stream.mesh_conf >= 0.95)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return int(np.argmin(np.abs(stream.yaw_deg)))
    return int(idx[np.argmin(score[idx])])


def photo_project_vertex_colors(geom: BFMGeometry, vertices_local: np.ndarray,
                                stream: Stream, controls: Controls,
                                color_model: ColorModel, fw: int, fh: int) -> Tuple[np.ndarray, Dict]:
    best = choose_best_frontal(stream)
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(best))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        rgba = np.tile(np.r_[color_model.skin_rgb, 1.0], (vertices_local.shape[0], 1))
        return np.clip(rgba * 255, 0, 255).astype(np.uint8), {"best_frame": best, "visible_vertex_ratio": 0.0, "fallback": "video_read_failed"}

    T_norm, scale, _center = build_T_for_frame(stream, controls, best, fw, fh)
    px, cam = project_vertices(vertices_local, T_norm, scale, fw, fh)
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = S[1, 1] = S[2, 2] = float(scale)
    R_node = (T_norm @ S)[:3, :3]
    normals = np.asarray(geom.base_normals, dtype=np.float64)
    ncam = (R_node @ normals.T).T

    xi = np.rint(px[:, 0]).astype(np.int32)
    yi = np.rint(px[:, 1]).astype(np.int32)
    inb = (xi >= 0) & (xi < fw) & (yi >= 0) & (yi < fh) & (cam[:, 2] < -0.5)
    front = ncam[:, 2] > 0.0
    if int((inb & front).sum()) < 1000:
        front = ncam[:, 2] < 0.0
    cand = inb & front
    pix = yi.astype(np.int64) * fw + xi.astype(np.int64)
    depth = -cam[:, 2].astype(np.float64)
    zbuf = np.full(fw * fh, np.inf, dtype=np.float64)
    cand_idx = np.where(cand)[0]
    np.minimum.at(zbuf, pix[cand_idx], depth[cand_idx])
    depth_tol = max(0.20, 0.055 * float(scale))
    visible = cand & (depth <= zbuf[pix] + depth_tol)

    rgba = np.tile(np.r_[color_model.skin_rgb, 1.0], (vertices_local.shape[0], 1)).astype(np.float32)
    rgb_frame = frame[:, :, ::-1].astype(np.float32) / 255.0
    rgba[visible, :3] = rgb_frame[yi[visible], xi[visible]]
    rgba[:, :3] = np.clip(rgba[:, :3], 0.0, 1.0)
    report = {
        "best_frame": int(best),
        "best_frame_yaw_deg": float(stream.yaw_deg[best]),
        "visible_vertices": int(visible.sum()),
        "total_vertices": int(vertices_local.shape[0]),
        "visible_vertex_ratio": float(visible.sum() / max(vertices_local.shape[0], 1)),
        "occluded_or_unprojected_fallback": "sampled skin tone",
        "method": "rounded-vertex z-buffer plus normal visibility; no invented texture",
    }
    return np.clip(rgba * 255.0, 0, 255).astype(np.uint8), report


def crop_head(img: np.ndarray, center: np.ndarray, scale_px: float,
              out_size: Tuple[int, int] = (220, 280)) -> np.ndarray:
    h, w = img.shape[:2]
    side = int(round(max(float(scale_px) * 2.35, 120.0)))
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1]) - 0.08 * side))
    x0 = cx - side // 2
    y0 = cy - side // 2
    x1 = x0 + side
    y1 = y0 + side
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.full((side, side, 3), 30, dtype=np.uint8)
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)


def old_generic_panel(frame_idx: int) -> np.ndarray:
    path = OLD_GENERIC_PROOFS.get(int(frame_idx), "")
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            return img[:, int(2 * w / 3):w]
    return np.full((PANEL_H, PANEL_W, 3), 35, dtype=np.uint8)


def add_cell_label(cell: np.ndarray, text: str) -> np.ndarray:
    out = cell.copy()
    draw_text(out, text, (8, 24), (255, 255, 255), 0.42, 1)
    return out


def encode_outputs(fps: float) -> Tuple[float, float]:
    subprocess.run([
        "ffmpeg", "-y", "-i", RAW_PATH,
        "-vcodec", "libx264", "-crf", "22", "-preset", "fast",
        "-movflags", "+faststart", MASTER_PATH,
    ], check=True, capture_output=True)

    attempts = [
        ("scale=720:-2", "32"),
        ("scale=640:-2", "35"),
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
    if os.path.exists(RAW_PATH):
        os.remove(RAW_PATH)
    return os.path.getsize(MASTER_PATH) / 1024**2, os.path.getsize(PREVIEW_PATH) / 1024**2


def preprocess_arcface_crop(crop_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(crop_bgr, (112, 112), interpolation=cv2.INTER_AREA)
    rgb = img[:, :, ::-1].astype(np.float32)
    blob = (rgb - 127.5) / 127.5
    return blob.transpose(2, 0, 1)[None, ...].astype(np.float32)


def arcface_embedding_onnx(crop_bgr: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    inp = session.get_inputs()[0].name
    out = session.run(None, {inp: preprocess_arcface_crop(crop_bgr)})[0].reshape(-1).astype(np.float32)
    return out / max(float(np.linalg.norm(out)), 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


def compute_arcface_report(crops: Dict[str, Dict[int, np.ndarray]]) -> Dict:
    model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
    report = {
        "method": "buffalo_l/w600k_r50.onnx direct 112x112 crop embedding fallback",
        "insightface_module_available": False,
        "model_path": model_path,
        "available": False,
    }
    try:
        import insightface  # type: ignore
        report["insightface_module_available"] = True
        report["insightface_version"] = str(getattr(insightface, "__version__", "unknown"))
    except Exception:
        pass
    if not os.path.exists(model_path):
        report["reason"] = "missing buffalo_l recognition model"
        return report
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        real0 = arcface_embedding_onnx(crops["real"][50], session)
        scores = {}
        for key in ["personalized_stylized", "personalized_photo", "generic_bfm", "old_generic_emoji", "personalized_shape_only", "generic_shape_only"]:
            if key in crops and 50 in crops[key]:
                scores[key] = cosine(real0, arcface_embedding_onnx(crops[key][50], session))
        real_sims = []
        for frame_idx, crop in crops.get("real", {}).items():
            if int(frame_idx) == 50:
                continue
            real_sims.append(cosine(real0, arcface_embedding_onnx(crop, session)))
        se = float(np.std(real_sims, ddof=1) / math.sqrt(len(real_sims))) if len(real_sims) >= 2 else float("nan")
        report.update({
            "available": True,
            "scores_vs_real_f50": scores,
            "between_real_frame_cosines": real_sims,
            "between_frame_se": se,
            "note": "Direct crop fallback is less strict than InsightFace FaceAnalysis alignment; use as supporting signal, not the human gate.",
        })
    except Exception as exc:
        report["reason"] = f"onnx_arcface_failed_{type(exc).__name__}"
    return report


def scan_scripts() -> Dict:
    terms = [
        ".".join(["torch", "cuda"]),
        "".join(["pytorch", "3d"]),
        "".join(["nvdi", "ffrast"]),
    ]
    hits = {}
    for term in terms:
        proc = subprocess.run(
            ["rg", "-n", term, "identity_bfm_solve_v1.py", "bfm_avatar_overlay_v1.py"],
            cwd=OUT_DIR,
            text=True,
            capture_output=True,
        )
        hits[term] = proc.stdout.strip().splitlines() if proc.returncode == 0 else []
    return {"hits": hits, "pass": all(len(v) == 0 for v in hits.values())}


def build_contact_sheet(crops: Dict[str, Dict[int, np.ndarray]]) -> None:
    cols = [
        ("real", "the real subject"),
        ("generic_bfm", "generic BFM"),
        ("personalized_stylized", "personalized stylized"),
        ("personalized_photo", "personalized photo"),
        ("old_generic_emoji", "old emoji"),
    ]
    cell_w, cell_h = 220, 280
    header_h = 34
    rows = []
    for frame_idx, label in CONTACT_FRAMES:
        cells = []
        for key, title in cols:
            cell = crops.get(key, {}).get(frame_idx)
            if cell is None:
                cell = np.full((cell_h, cell_w, 3), 35, dtype=np.uint8)
            cell = add_cell_label(cell, title if key != "real" else f"{label} f{frame_idx}")
            cells.append(cell)
        rows.append(np.hstack(cells))
    sheet = np.vstack(rows)
    cv2.imwrite(CONTACT_SHEET_PATH, sheet)


def write_notes(report: Dict) -> None:
    verdict = report["likeness_verdict"]
    lines = [
        "# BFM Avatar v1 Notes",
        "",
        "## Outputs",
        f"- 3-up master: `{MASTER_PATH}`",
        f"- Preview: `{PREVIEW_PATH}`",
        f"- Contact sheet: `{CONTACT_SHEET_PATH}`",
        f"- Identity: `{IDENTITY_PATH}`",
        f"- JSON report: `{REPORT_PATH}`",
        "",
        "## Identity And License",
        f"- Accepted identity frames: {report['identity']['accepted_count']}",
        f"- Identity stability p90: {report['identity']['stability_p90']:.4f}",
        "- BFM asset license flag: academic-use-only (`_deps/3DDFA_V2/bfm/readme.md`); this is local proof-only, not commercial-clean.",
        "",
        "## Render",
        f"- Alpha coverage: {report['render_metrics']['alpha_coverage_pct']:.2f}%",
        f"- Center error p95/head-scale: {report['render_metrics']['center_error_over_head_scale']['p95']:.4f}",
        f"- BBox ratio p95: {report['render_metrics']['bbox_ratio_over_head_scale']['p95']:.4f}",
        f"- Source transition pop max/head-scale: {report['render_metrics']['source_transition_pop_over_head_scale']['max']:.4f}",
        "",
        "## Likeness Verdict",
        f"- Verdict: {verdict['verdict']}",
        f"- Better variant: {verdict['better_variant']}",
        f"- Limit: {verdict['limit']}",
        "",
        "## Honesty",
        "- Stylized variant uses sampled skin/feature colors plus simple vertex-color feature regions; it is not a face swap.",
        "- Photo variant samples visible vertices from one frontal frame and falls back to sampled skin where vertices are unobserved or occluded.",
        "- Profile frames are amber-tinted; interpolated/held frames are blue-tinted.",
        "- The human contact sheet remains the primary gate for whether it reads as the subject.",
    ]
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run() -> Dict:
    t0 = time.time()
    fps, fw, fh, video_n = video_meta()
    stream = load_stream()
    controls = load_controls()
    ident = np.load(IDENTITY_PATH, allow_pickle=True)
    alpha_shp = np.asarray(ident["alpha_shp"], dtype=np.float32)

    print(f"{LOG_PREFIX} loading BFM geometry")
    geom = load_bfm_geometry(alpha_shp)
    color_model = build_color_model(stream, geom)
    expr = fit_expression_track(stream, controls, fw, fh)
    alpha_exp = np.asarray(expr["alpha_exp"], dtype=np.float32)

    photo_rgba, photo_report = photo_project_vertex_colors(
        geom,
        vertices_with_expression(geom.identity_base_local, geom.w_exp_local, alpha_exp[choose_best_frontal(stream)]),
        stream,
        controls,
        color_model,
        fw,
        fh,
    )

    renderer = BFMRenderer(fw, fh, FOCAL_LEN)
    cap = cv2.VideoCapture(VIDEO_PATH)
    writer = cv2.VideoWriter(RAW_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (PANEL_W * 3, PANEL_H))

    alpha_frames = 0
    center_errors: List[float] = []
    bbox_ratios: List[float] = []
    bbox_centers = np.full((len(stream.frame), 2), np.nan, dtype=np.float32)
    bfm_apertures = np.full(len(stream.frame), np.nan, dtype=np.float32)
    mesh_apertures = np.asarray([mouth_aperture_stream(stream.projected_px[i]) for i in range(len(stream.frame))], dtype=np.float32)
    saved: Dict[int, Dict[str, np.ndarray]] = {}
    crops: Dict[str, Dict[int, np.ndarray]] = {
        "real": {},
        "generic_bfm": {},
        "personalized_stylized": {},
        "personalized_photo": {},
        "old_generic_emoji": {},
        "personalized_shape_only": {},
        "generic_shape_only": {},
    }

    print(f"{LOG_PREFIX} rendering {len(stream.frame)} frames")
    for i in range(len(stream.frame)):
        ok, frame = cap.read()
        if not ok:
            break
        T_norm, av_scale, center = build_T_for_frame(stream, controls, i, fw, fh)
        verts = vertices_with_expression(geom.identity_base_local, geom.w_exp_local, alpha_exp[i])

        rgba_sty = renderer.render(verts, geom.faces, T_norm, av_scale, color_model.stylized_vertex_rgba)
        rgba_photo = renderer.render(verts, geom.faces, T_norm, av_scale, photo_rgba)
        rgba_sty = tint_rgba_by_source(rgba_sty, stream.mesh_source[i])
        rgba_photo = tint_rgba_by_source(rgba_photo, stream.mesh_source[i])
        out_sty = composite_rgba_bgr(frame, rgba_sty)
        out_photo = composite_rgba_bgr(frame, rgba_photo)

        bbox = alpha_bbox(rgba_sty)
        if bbox is not None:
            alpha_frames += 1
            x0, y0, x1, y1 = bbox
            bc = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            bbox_centers[i] = bc
            center_errors.append(float(np.linalg.norm(bc - center) / max(float(controls.head_scale_px[i]), 1.0)))
            bbox_ratios.append(float(max(x1 - x0 + 1, y1 - y0 + 1) / max(float(controls.head_scale_px[i]), 1.0)))
        bfm_apertures[i] = mouth_aperture_bfm(geom, verts)

        draw_text(out_sty, f"f{i:04d} {stream.mesh_source[i]} yaw={stream.yaw_deg[i]:+.0f}", (12, 32), (70, 255, 150), 0.55, 1)
        draw_text(out_photo, f"photo vertices visible={photo_report['visible_vertex_ratio']:.2f}", (12, 32), (70, 220, 255), 0.50, 1)
        writer.write(panelize(frame, out_sty, out_photo))

        if i in [f for f, _label in CONTACT_FRAMES] or i == 485:
            rgba_gen = renderer.render(
                vertices_with_expression(geom.generic_base_local, geom.w_exp_local, alpha_exp[i]),
                geom.faces,
                T_norm,
                av_scale,
                color_model.generic_vertex_rgba,
            )
            rgba_shape = renderer.render(geom.identity_base_local, geom.faces, T_norm, av_scale, None, np.asarray([0.62, 0.48, 0.38, 1.0], dtype=np.float32))
            rgba_gen_shape = renderer.render(geom.generic_base_local, geom.faces, T_norm, av_scale, None, np.asarray([0.62, 0.62, 0.62, 1.0], dtype=np.float32))
            out_gen = composite_rgba_bgr(frame, tint_rgba_by_source(rgba_gen, stream.mesh_source[i]))
            out_shape = composite_rgba_bgr(frame, rgba_shape)
            out_gen_shape = composite_rgba_bgr(frame, rgba_gen_shape)
            saved[i] = {"real": frame.copy(), "stylized": out_sty.copy(), "photo": out_photo.copy(), "generic": out_gen.copy()}
            center_i = np.asarray(controls.center_px[i], dtype=np.float32)
            scale_i = float(controls.head_scale_px[i])
            if i in [f for f, _label in CONTACT_FRAMES]:
                crops["real"][i] = crop_head(frame, center_i, scale_i)
                crops["generic_bfm"][i] = crop_head(out_gen, center_i, scale_i)
                crops["personalized_stylized"][i] = crop_head(out_sty, center_i, scale_i)
                crops["personalized_photo"][i] = crop_head(out_photo, center_i, scale_i)
                old_panel = old_generic_panel(i)
                crops["old_generic_emoji"][i] = cv2.resize(old_panel, (220, 280), interpolation=cv2.INTER_AREA)
                crops["personalized_shape_only"][i] = crop_head(out_shape, center_i, scale_i)
                crops["generic_shape_only"][i] = crop_head(out_gen_shape, center_i, scale_i)
                cv2.imwrite(f"{OUT_DIR}/subject_bfm_v1_proof_{i:04d}_stylized.jpg", out_sty)
                cv2.imwrite(f"{OUT_DIR}/subject_bfm_v1_proof_{i:04d}_photo.jpg", out_photo)
        if i % 50 == 0:
            elapsed = time.time() - t0
            fps_proc = (i + 1) / max(elapsed, 1e-6)
            eta = (len(stream.frame) - i - 1) / max(fps_proc, 1e-6)
            print(f"{LOG_PREFIX} f{i:04d}/{len(stream.frame)} {fps_proc:.2f}fps ETA {eta:.0f}s alpha={100*alpha_frames/max(i+1,1):.1f}%")

    cap.release()
    writer.release()
    renderer.close()

    build_contact_sheet(crops)
    master_mb, preview_mb = encode_outputs(fps)

    transition_jumps = []
    transition_rows = []
    for i in range(1, len(stream.frame)):
        if str(stream.mesh_source[i]) == str(stream.mesh_source[i - 1]):
            continue
        if not (np.isfinite(bbox_centers[i]).all() and np.isfinite(bbox_centers[i - 1]).all()):
            continue
        jump = float(np.linalg.norm(bbox_centers[i] - bbox_centers[i - 1]) / max(float(controls.head_scale_px[i]), 1.0))
        transition_jumps.append(jump)
        transition_rows.append({
            "frame": int(i),
            "from": str(stream.mesh_source[i - 1]),
            "to": str(stream.mesh_source[i]),
            "jump_over_head_scale": jump,
        })

    frontal_mask = stream.geometry_observed & (np.abs(stream.yaw_deg) < 20.0)
    observed_non_profile = stream.geometry_observed & (np.abs(stream.yaw_deg) < 75.0)
    expr_metrics = {
        "bfm_aperture_vs_mesh_aperture_r_frontal": pearsonr_np(bfm_apertures[frontal_mask], mesh_apertures[frontal_mask]),
        "bfm_aperture_vs_mesh_aperture_r_observed_non_profile": pearsonr_np(bfm_apertures[observed_non_profile], mesh_apertures[observed_non_profile]),
        "frame_485": {
            "bfm_aperture": float(bfm_apertures[485]) if len(bfm_apertures) > 485 else float("nan"),
            "mesh_aperture": float(mesh_apertures[485]) if len(mesh_apertures) > 485 else float("nan"),
            "source": str(stream.mesh_source[485]) if len(stream.mesh_source) > 485 else "",
            "yaw_deg": float(stream.yaw_deg[485]) if len(stream.yaw_deg) > 485 else float("nan"),
        },
    }

    arcface = compute_arcface_report(crops)
    identity_report = json.load(open(f"{OUT_DIR}/subject_bfm_identity_v1_report.json", "r", encoding="utf-8"))
    scan = scan_scripts()

    render_metrics = {
        "alpha_coverage_pct": float(100.0 * alpha_frames / max(len(stream.frame), 1)),
        "center_error_over_head_scale": summarize_array(center_errors),
        "bbox_ratio_over_head_scale": summarize_array(bbox_ratios),
        "source_transition_pop_over_head_scale": {
            **summarize_array(transition_jumps),
            "transitions": transition_rows,
        },
    }

    verdict = {
        "verdict": "partial stylized resemblance; not converged on identity shape",
        "better_variant": "stylized",
        "limit": (
            "Contact sheet shows hair/beard/color cues help, but personalized BFM shape remains visually close "
            "to generic BFM. Held-out sparse reprojection and ArcFace fallback do not clear the generic-BFM "
            "baseline; photo projection has low visible-vertex fill and is comparison-only."
        ),
    }

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_unix": time.time(),
        "paths": {
            "identity": IDENTITY_PATH,
            "expression_cache": EXPR_CACHE_PATH,
            "master": MASTER_PATH,
            "preview": PREVIEW_PATH,
            "contact_sheet": CONTACT_SHEET_PATH,
            "notes": NOTES_PATH,
            "report": REPORT_PATH,
        },
        "output_sizes_mb": {
            "master": round(master_mb, 3),
            "preview": round(preview_mb, 3),
            "contact_sheet": round(os.path.getsize(CONTACT_SHEET_PATH) / 1024**2, 3),
        },
        "identity": {
            "accepted_count": int(identity_report["frame_selection"]["accepted_count"]),
            "stability_p90": float(identity_report["identity_stability"]["p90"]),
            "heldout_reprojection": identity_report["fixed_pose_reprojection"]["heldout"],
            "license_flag": identity_report["license_flag"],
        },
        "bfm_mesh": {
            "vertices": int(geom.identity_base_local.shape[0]),
            "faces": int(geom.faces.shape[0]),
            "dropped_out_of_range_faces": int(geom.dropped_face_count),
            "normalization": {
                "center_model": geom.center_model.tolist(),
                "scale_model": geom.scale_model,
            },
        },
        "color_model": {
            "skin_rgb": color_model.skin_rgb.tolist(),
            "brow_rgb": color_model.brow_rgb.tolist(),
            "hair_rgb": color_model.hair_rgb.tolist(),
            "facial_rgb": color_model.facial_rgb.tolist(),
            "lip_rgb": color_model.lip_rgb.tolist(),
            "feature_enabled": color_model.feature_enabled,
            "sample_counts": color_model.sample_counts,
            "sample_std": color_model.sample_std,
            "mask_counts": color_model.mask_counts,
        },
        "photo_projection": photo_report,
        "expression": {
            "track_report": expr["report"],
            "metrics": expr_metrics,
            "source_counts": {str(k): int(v) for k, v in zip(*np.unique(expr["expr_source"], return_counts=True))},
        },
        "render_metrics": render_metrics,
        "arcface": arcface,
        "success_checks": {
            "identity_stability_p90_le_0p25": bool(float(identity_report["identity_stability"]["p90"]) <= 0.25),
            "heldout_reprojection_beats_generic": bool(identity_report["fixed_pose_reprojection"]["heldout"].get("personalized_beats_generic_mean", False)),
            "alpha_ge_99pct": bool(render_metrics["alpha_coverage_pct"] >= 99.0),
            "center_p95_le_0p15": bool(render_metrics["center_error_over_head_scale"]["p95"] <= 0.15),
            "bbox_p95_in_0p70_1p35": bool(0.70 <= render_metrics["bbox_ratio_over_head_scale"]["p95"] <= 1.35),
            "source_transition_pop_le_0p12": bool(render_metrics["source_transition_pop_over_head_scale"]["max"] <= 0.12),
            "jaw_r_frontal_ge_0p85": bool(expr_metrics["bfm_aperture_vs_mesh_aperture_r_frontal"] >= 0.85),
            "jaw_r_observed_non_profile_ge_0p75": bool(expr_metrics["bfm_aperture_vs_mesh_aperture_r_observed_non_profile"] >= 0.75),
            "f485_mouth_open": bool(expr_metrics["frame_485"]["bfm_aperture"] > np.nanpercentile(bfm_apertures, 75.0)),
            "disallowed_renderer_scan_clean": bool(scan["pass"]),
        },
        "mps_disallowed_scan": scan,
        "likeness_verdict": verdict,
        "honest_limits": [
            "Bundled BFM asset is academic-use-only.",
            "Photo projection contains only pixels visible in the selected frontal frame; unobserved vertices fall back to sampled skin tone.",
            "Stylized feature regions are coarse vertex-color masks, not a face swap.",
            "Profile and interpolated frames carry source-honesty tinting.",
            "InsightFace Python package is not required here; the report uses the local buffalo_l recognition ONNX as a fallback if available.",
        ],
        "processing_time_s": round(time.time() - t0, 3),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_notes(report)

    print(f"{LOG_PREFIX} master {MASTER_PATH} ({master_mb:.2f} MB)")
    print(f"{LOG_PREFIX} preview {PREVIEW_PATH} ({preview_mb:.2f} MB)")
    print(f"{LOG_PREFIX} contact sheet {CONTACT_SHEET_PATH}")
    print(f"{LOG_PREFIX} center_p95={render_metrics['center_error_over_head_scale']['p95']:.4f} bbox_p95={render_metrics['bbox_ratio_over_head_scale']['p95']:.4f}")
    print(f"{LOG_PREFIX} expression r frontal={expr_metrics['bfm_aperture_vs_mesh_aperture_r_frontal']:.3f}")
    return report


if __name__ == "__main__":
    run()
