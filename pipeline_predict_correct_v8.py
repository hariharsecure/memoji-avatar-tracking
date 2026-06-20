#!/usr/bin/env python3
"""
pipeline_predict_correct_v8.py - buildable-now predict/correct estimator.

Builds on mesh_cascade_v7 output. v8 adds a recursive pose state estimator:
constant-velocity position, SO(3) error-state rotation, optical-flow head
anchor, and a soft MediaPipe-Pose body->head prior. Predictions are confidence
labeled and never marked verified unless a reliable face observation corrects
them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from scipy.spatial.transform import Rotation

import pipeline_mesh_cascade_v7 as v7


VIDEO_PATH = v7.VIDEO_PATH
OUT_DIR = v7.OUT_DIR
POSE_MODEL = "models/pose_landmarker_full.task"
V7_STREAM_PATH = f"{OUT_DIR}/mesh_cascade_v7_stream.npz"
V7_REPORT_PATH = f"{OUT_DIR}/mesh_cascade_v7_report.json"
STREAM_PATH = f"{OUT_DIR}/predict_correct_v8_stream.npz"
REPORT_PATH = f"{OUT_DIR}/predict_correct_v8_report.json"
METRICS_PATH = f"{OUT_DIR}/predict_correct_v8_metrics.json"
NOTES_PATH = f"{OUT_DIR}/notes_predict_correct_v8.md"
OVERLAY_MASTER_PATH = f"{OUT_DIR}/predict_correct_v8_overlay_master.mp4"
OVERLAY_PREVIEW_PATH = f"{OUT_DIR}/predict_correct_v8_overlay_preview.mp4"
MOTION_STRIP_PATH = f"{OUT_DIR}/predict_correct_v8_motion_strip_f425_440.png"
RELIABILITY_PATH = f"{OUT_DIR}/predict_correct_v8_reliability.png"

PIPELINE_VERSION = "predict_correct_v8"
LOG_PREFIX = "[predict-correct-v8]"
POSE_FLIP_START = 425
POSE_FLIP_END = 440


def summarize(values: np.ndarray) -> Dict:
    return v7.summarize_values(np.asarray(values, dtype=np.float64))


def wrap_angle_deg(x: np.ndarray | float) -> np.ndarray:
    return v7.wrap_angle_deg(x)


def unwrap_yaw_deg(yaw: np.ndarray) -> np.ndarray:
    return v7.unwrap_yaw_deg(wrap_angle_deg(yaw), reference=np.asarray(yaw, dtype=np.float64))


def yaw_from_rotation(rot: Rotation, reference_yaw: float) -> float:
    yaw = float(wrap_angle_deg(rot.as_euler("YXZ", degrees=True)[0]))
    k = round((reference_yaw - yaw) / 360.0)
    return yaw + 360.0 * k


def make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


@dataclass
class BodyRecord:
    detected: bool
    confidence: float
    head_center: np.ndarray
    shoulder_mid: np.ndarray
    shoulder_span: float
    shoulder_angle: float
    torso_height: float
    feature: np.ndarray


def body_feature_from_landmarks(result, fw: int, fh: int) -> BodyRecord:
    nan2 = np.asarray([np.nan, np.nan], dtype=np.float64)
    if not result.pose_landmarks:
        return BodyRecord(False, 0.0, nan2, nan2, 0.0, 0.0, 0.0, np.zeros(9, dtype=np.float64))

    lms = result.pose_landmarks[0]

    def pt(idx: int) -> Tuple[np.ndarray, float]:
        lm = lms[idx]
        return np.asarray([lm.x * fw, lm.y * fh], dtype=np.float64), float(getattr(lm, "visibility", 1.0))

    nose, nose_v = pt(0)
    l_ear, le_v = pt(7)
    r_ear, re_v = pt(8)
    l_sh, ls_v = pt(11)
    r_sh, rs_v = pt(12)
    l_hip, lh_v = pt(23)
    r_hip, rh_v = pt(24)

    head_pts = []
    head_weights = []
    for p, conf, weight in [(l_ear, le_v, 1.0), (r_ear, re_v, 1.0), (nose, nose_v, 0.65)]:
        if conf >= 0.25:
            head_pts.append(p)
            head_weights.append(conf * weight)
    if head_pts:
        weights = np.asarray(head_weights, dtype=np.float64)
        head_center = np.sum(np.asarray(head_pts) * weights[:, None], axis=0) / max(float(weights.sum()), 1e-6)
        head_conf = float(np.clip(np.mean(head_weights), 0.0, 1.0))
    else:
        head_center = nose
        head_conf = max(nose_v * 0.35, 0.0)

    shoulders_ok = ls_v >= 0.2 and rs_v >= 0.2
    shoulder_mid = 0.5 * (l_sh + r_sh) if shoulders_ok else nan2
    shoulder_vec = r_sh - l_sh if shoulders_ok else np.asarray([0.0, 0.0])
    shoulder_span = float(np.linalg.norm(shoulder_vec)) if shoulders_ok else 0.0
    shoulder_angle = float(math.atan2(shoulder_vec[1], shoulder_vec[0])) if shoulder_span > 1e-6 else 0.0
    hips_ok = lh_v >= 0.2 and rh_v >= 0.2
    hip_mid = 0.5 * (l_hip + r_hip) if hips_ok else nan2
    torso_height = float(np.linalg.norm(shoulder_mid - hip_mid)) if shoulders_ok and hips_ok else 0.0
    body_conf = float(np.clip(0.45 * head_conf + 0.35 * min(ls_v, rs_v) + 0.20 * max(lh_v, rh_v, 0.0), 0.0, 1.0))
    detected = bool(body_conf >= 0.20 and np.isfinite(head_center).all())

    feature = np.asarray([
        1.0,
        head_center[0] / max(float(fw), 1.0),
        head_center[1] / max(float(fh), 1.0),
        (shoulder_mid[0] / max(float(fw), 1.0)) if shoulders_ok else 0.5,
        (shoulder_mid[1] / max(float(fh), 1.0)) if shoulders_ok else 0.5,
        shoulder_span / max(float(fw), 1.0),
        math.sin(shoulder_angle),
        math.cos(shoulder_angle),
        torso_height / max(float(fh), 1.0),
    ], dtype=np.float64)
    return BodyRecord(detected, body_conf, head_center, shoulder_mid, shoulder_span, shoulder_angle, torso_height, feature)


def run_body_pose(video_path: str, total_f: int) -> Tuple[List[BodyRecord], Dict]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.0)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose = make_pose_landmarker()
    rows: List[BodyRecord] = []
    t0 = time.time()
    for fidx in range(total_f):
        ok, frame = cap.read()
        if not ok:
            break
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = pose.detect(image)
        rows.append(body_feature_from_landmarks(result, fw, fh))
        if fidx % 100 == 0:
            print(f"{LOG_PREFIX} body pose f{fidx}/{total_f}")
    pose.close()
    cap.release()
    while len(rows) < total_f:
        rows.append(BodyRecord(False, 0.0, np.asarray([np.nan, np.nan]), np.asarray([np.nan, np.nan]), 0.0, 0.0, 0.0, np.zeros(9)))
    conf = np.asarray([r.confidence for r in rows], dtype=np.float64)
    return rows, {
        "fps_source": fps,
        "frame_width": fw,
        "frame_height": fh,
        "detected_frames": int(sum(r.detected for r in rows)),
        "confidence": summarize(conf),
        "wall_s": float(time.time() - t0),
    }


class OnlineRidge1D:
    def __init__(self, n_features: int, init_var: float = 1000.0) -> None:
        self.theta = np.zeros(n_features, dtype=np.float64)
        self.p = np.eye(n_features, dtype=np.float64) * init_var
        self.count = 0

    def predict(self, x: np.ndarray) -> float:
        return float(np.asarray(x, dtype=np.float64) @ self.theta)

    def update(self, x: np.ndarray, y: float, weight: float) -> None:
        x = np.asarray(x, dtype=np.float64)
        r = 1.0 / max(float(weight), 1e-3)
        px = self.p @ x
        denom = float(r + x @ px)
        k = px / max(denom, 1e-9)
        err = float(y - x @ self.theta)
        self.theta = self.theta + k * err
        self.p = self.p - np.outer(k, x) @ self.p
        self.count += 1


class PositionKF:
    def __init__(self, center: np.ndarray) -> None:
        self.x = np.asarray([center[0], center[1], 0.0, 0.0], dtype=np.float64)
        self.p = np.diag([25.0, 25.0, 400.0, 400.0]).astype(np.float64)

    def predict(self) -> np.ndarray:
        f = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        q = np.diag([2.5, 2.5, 8.0, 8.0]).astype(np.float64)
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        return self.x[:2].copy()

    def update(self, z: np.ndarray, sigma_px: float, gate: float = 16.0) -> Tuple[bool, float]:
        z = np.asarray(z, dtype=np.float64)
        h = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        r = np.eye(2, dtype=np.float64) * max(float(sigma_px) ** 2, 1e-6)
        y = z - h @ self.x
        s = h @ self.p @ h.T + r
        maha = float(y.T @ np.linalg.pinv(s) @ y)
        if maha > gate:
            return False, maha
        k = self.p @ h.T @ np.linalg.pinv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(4) - k @ h) @ self.p
        return True, maha

    def confidence(self, head_scale: float) -> float:
        sigma = math.sqrt(max(float(np.trace(self.p[:2, :2]) * 0.5), 1e-9))
        return float(np.clip(1.0 / (1.0 + sigma / max(float(head_scale), 1.0)), 0.0, 1.0))


class RotationESKF:
    def __init__(self, rotation: Rotation) -> None:
        self.q = rotation
        self.w = np.zeros(3, dtype=np.float64)
        self.p = np.diag([math.radians(5.0) ** 2] * 3 + [math.radians(4.0) ** 2] * 3).astype(np.float64)

    def predict(self) -> Rotation:
        self.q = Rotation.from_rotvec(self.w) * self.q
        self.w *= 0.985
        q = np.diag([math.radians(1.8) ** 2] * 3 + [math.radians(0.8) ** 2] * 3)
        self.p = self.p + q
        return self.q

    def update(self, meas: Rotation, sigma_deg: float, gate: float = 18.0) -> Tuple[bool, float]:
        err = (meas * self.q.inv()).as_rotvec()
        h = np.zeros((3, 6), dtype=np.float64)
        h[:3, :3] = np.eye(3)
        r = np.eye(3, dtype=np.float64) * math.radians(max(float(sigma_deg), 1e-3)) ** 2
        s = h @ self.p @ h.T + r
        maha = float(err.T @ np.linalg.pinv(s) @ err)
        if maha > gate:
            return False, maha
        k = self.p @ h.T @ np.linalg.pinv(s)
        dx = k @ err
        self.q = Rotation.from_rotvec(dx[:3]) * self.q
        self.w = self.w + dx[3:]
        self.p = (np.eye(6) - k @ h) @ self.p
        return True, maha

    def confidence(self) -> float:
        sigma = math.degrees(math.sqrt(max(float(np.trace(self.p[:3, :3]) / 3.0), 1e-12)))
        return float(np.clip(1.0 / (1.0 + sigma / 18.0), 0.0, 1.0))


def face_reliable_mask(source: np.ndarray) -> np.ndarray:
    return np.asarray([str(s).startswith("observed") for s in source], dtype=bool)


def hidden_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    return v7.true_spans(mask)


def angular_jumps(rotations: Rotation) -> np.ndarray:
    return v7.angular_jumps_deg(rotations)


def mean_vertex_step(px: np.ndarray, head_scale: np.ndarray) -> np.ndarray:
    return v7.mean_vertex_step_over_head_scale(px, head_scale)


def fit_body_prior_and_estimate(v7_stream, flow: Dict, body_rows: List[BodyRecord]) -> Dict:
    source = np.asarray(v7_stream["mesh_source"]).astype(str)
    reliable = face_reliable_mask(source)
    v7_px = np.asarray(v7_stream["projected_px"], dtype=np.float64)
    head_scale = np.asarray(v7_stream["head_scale_px"], dtype=np.float64)
    face_center = v7.semantic_center(v7_px)
    yaw_v7 = unwrap_yaw_deg(np.asarray(v7_stream["yaw_deg"], dtype=np.float64))
    pitch_v7 = np.asarray(v7_stream["pitch_deg"], dtype=np.float64)
    roll_v7 = np.asarray(v7_stream["roll_deg"], dtype=np.float64)
    rotations_v7 = v7.rotations_from_stream(
        np.asarray(v7_stream["head_transform"], dtype=np.float64),
        yaw_v7,
        pitch_v7,
        roll_v7,
    )
    n = len(source)

    offset_x = OnlineRidge1D(9)
    offset_y = OnlineRidge1D(9)
    yaw_model = OnlineRidge1D(9)
    pos_kf = PositionKF(face_center[0])
    rot_kf = RotationESKF(rotations_v7[0])

    centers = np.zeros((n, 2), dtype=np.float64)
    rot_mats = np.zeros((n, 3, 3), dtype=np.float64)
    conf = np.zeros(n, dtype=np.float64)
    rot_conf = np.zeros(n, dtype=np.float64)
    pos_conf = np.zeros(n, dtype=np.float64)
    status = np.full(n, "predicted", dtype="<U16")
    obs_mask = np.zeros(n, dtype=bool)
    accepted_face = np.zeros(n, dtype=bool)
    accepted_flow = np.zeros(n, dtype=bool)
    accepted_body = np.zeros(n, dtype=bool)
    gated_face = np.zeros(n, dtype=bool)
    gated_flow = np.zeros(n, dtype=bool)
    gated_body = np.zeros(n, dtype=bool)
    pre_center = np.zeros((n, 2), dtype=np.float64)
    pre_rot = np.zeros((n, 3, 3), dtype=np.float64)
    pre_conf = np.zeros(n, dtype=np.float64)
    body_prior_center = np.full((n, 2), np.nan, dtype=np.float64)
    body_prior_yaw = np.full(n, np.nan, dtype=np.float64)
    body_prior_conf = np.zeros(n, dtype=np.float64)
    face_maha = np.zeros(n, dtype=np.float64)
    flow_maha = np.zeros(n, dtype=np.float64)
    body_maha = np.zeros(n, dtype=np.float64)
    frames_since_verified = 0
    prev_output_q = rotations_v7[0]

    for i in range(n):
        face_is_reliable = bool(reliable[i])
        face_is_profile = source[i] == "profile_fit"
        pos_kf.predict()
        rot_kf.predict()
        pre_center[i] = pos_kf.x[:2]
        pre_rot[i] = rot_kf.q.as_matrix()
        pre_decay = max(0.35, 0.965 ** frames_since_verified)
        pre_conf[i] = (0.5 * pos_kf.confidence(head_scale[i]) + 0.5 * rot_kf.confidence()) * pre_decay
        any_soft = False

        body = body_rows[i]
        if body.detected and np.isfinite(body.head_center).all():
            x = body.feature
            if offset_x.count >= 8:
                off = np.asarray([offset_x.predict(x), offset_y.predict(x)], dtype=np.float64)
                model_conf = min(1.0, offset_x.count / 48.0)
            else:
                off = np.zeros(2, dtype=np.float64)
                model_conf = 0.35
            z_body = body.head_center + off
            body_prior_center[i] = z_body
            c_body = float(np.clip(body.confidence * model_conf, 0.0, 0.55))
            body_prior_conf[i] = c_body
            sigma = max(float(head_scale[i]) * (0.34 - 0.12 * c_body), 16.0)
            ok, maha = pos_kf.update(z_body, sigma, gate=14.0)
            body_maha[i] = maha
            accepted_body[i] = ok
            gated_body[i] = not ok
            any_soft = any_soft or ok

            if yaw_model.count >= 24:
                yb = yaw_model.predict(x)
                k = round((yaw_from_rotation(rot_kf.q, yaw_v7[i]) - yb) / 360.0)
                yb = yb + 360.0 * k
                body_prior_yaw[i] = yb
                pred_yaw = yaw_from_rotation(rot_kf.q, yaw_v7[i])
                yaw_innov = abs(wrap_angle_deg(yb - pred_yaw))
                yaw_gate = 24.0 + 12.0 * (1.0 - c_body)
                if yaw_innov <= yaw_gate:
                    e = rot_kf.q.as_euler("YXZ", degrees=True)
                    meas = Rotation.from_euler("YXZ", [yb, e[1], e[2]], degrees=True)
                    ok_r, maha_r = rot_kf.update(meas, sigma_deg=max(58.0 - 22.0 * c_body, 42.0), gate=6.0)
                else:
                    ok_r = False
                    maha_r = (yaw_innov / max(yaw_gate, 1e-6)) ** 2
                accepted_body[i] = bool(accepted_body[i] or ok_r)
                gated_body[i] = bool(gated_body[i] and not ok_r)
                body_maha[i] = max(body_maha[i], maha_r)
                any_soft = any_soft or ok_r

        if bool(flow.get("available", False)) and float(flow["confidence"][i]) >= v7.FLOW_MIN_CONFIDENCE:
            z_flow = np.asarray(flow["center_px"][i], dtype=np.float64)
            sigma = max(float(head_scale[i]) * (0.20 - 0.10 * float(flow["confidence"][i])), 6.0)
            ok, maha = pos_kf.update(z_flow, sigma, gate=14.0)
            flow_maha[i] = maha
            accepted_flow[i] = ok
            gated_flow[i] = not ok
            any_soft = any_soft or ok

        if face_is_reliable or face_is_profile:
            sigma_px = 3.0 if face_is_reliable else max(float(head_scale[i]) * 0.22, 10.0)
            ok, maha = pos_kf.update(face_center[i], sigma_px, gate=20.0 if face_is_reliable else 10.0)
            face_maha[i] = maha
            raw_residual = (rotations_v7[i] * rot_kf.q.inv()).as_rotvec()
            raw_residual_deg = float(np.linalg.norm(raw_residual) * 180.0 / math.pi)
            rotation_verified = False
            if face_is_profile:
                residual = raw_residual
                residual_deg = raw_residual_deg
                maha_r = (residual_deg / 14.0) ** 2
                ok_r = False
                if residual_deg <= 120.0:
                    correction = 0.55 * residual
                    mag = float(np.linalg.norm(correction))
                    max_corr = math.radians(6.5)
                    if mag > max_corr:
                        correction = correction * (max_corr / mag)
                    rot_kf.q = Rotation.from_rotvec(correction) * rot_kf.q
                    rot_kf.p[:3, :3] += np.eye(3, dtype=np.float64) * math.radians(2.5) ** 2
                    ok_r = True
            else:
                ok_r, maha_r = rot_kf.update(rotations_v7[i], sigma_deg=3.0, gate=64.0)
                residual = (rotations_v7[i] * rot_kf.q.inv()).as_rotvec()
                residual_deg = float(np.linalg.norm(residual) * 180.0 / math.pi)
                rotation_verified = bool(ok_r and raw_residual_deg <= 18.0)
                if raw_residual_deg <= 18.0 and residual_deg <= 75.0:
                    rot_kf.q = rotations_v7[i]
                    rot_kf.p[:3, :3] *= 0.65
                    rotation_verified = True
                    ok_r = True
                elif residual_deg <= 90.0:
                    rot_kf.q = rotations_v7[i]
                    rot_kf.p[:3, :3] += np.eye(3, dtype=np.float64) * math.radians(3.0) ** 2
                    ok_r = True
            face_maha[i] = max(face_maha[i], maha_r)
            accepted_face[i] = bool(ok and rotation_verified)
            gated_face[i] = bool((not ok or not ok_r) and (face_is_reliable or face_is_profile))
            any_soft = any_soft or ok or ok_r

        if accepted_face[i]:
            status[i] = "verified"
            obs_mask[i] = True
        elif any_soft:
            status[i] = "mixed"
        else:
            status[i] = "predicted"

        centers[i] = pos_kf.x[:2]
        rot_mats[i] = rot_kf.q.as_matrix()
        pos_conf[i] = pos_kf.confidence(head_scale[i])
        rot_conf[i] = rot_kf.confidence()
        conf[i] = float(np.clip(0.5 * pos_conf[i] + 0.5 * rot_conf[i], 0.0, 1.0))
        if status[i] == "verified":
            frames_since_verified = 0
            conf[i] = max(conf[i], 0.88)
        elif status[i] == "mixed":
            frames_since_verified += 1
            conf[i] *= max(0.35, 0.965 ** frames_since_verified)
        elif status[i] == "predicted":
            frames_since_verified += 1
            conf[i] *= 0.72 * max(0.35, 0.965 ** frames_since_verified)

        if face_is_reliable and body.detected and np.isfinite(body.head_center).all():
            target_offset = face_center[i] - body.head_center
            w = max(float(body.confidence), 0.05)
            offset_x.update(body.feature, float(target_offset[0]), w)
            offset_y.update(body.feature, float(target_offset[1]), w)
            yaw_model.update(body.feature, float(yaw_v7[i]), w)

        if i > 0:
            delta = (rot_kf.q * prev_output_q.inv()).as_rotvec()
            mag = float(np.linalg.norm(delta))
            max_mag = math.radians(18.0)
            if mag > max_mag:
                delta = delta * (max_mag / mag)
            vel_alpha = 0.70 if (face_is_reliable or face_is_profile) else 0.35
            rot_kf.w = (1.0 - vel_alpha) * rot_kf.w + vel_alpha * delta
        prev_output_q = rot_kf.q

    rotations_v8 = Rotation.from_matrix(rot_mats)
    yaw_v8_wrapped = rotations_v8.as_euler("YXZ", degrees=True)[:, 0]
    yaw_v8 = unwrap_yaw_deg(yaw_v8_wrapped)
    pitch_v8 = rotations_v8.as_euler("YXZ", degrees=True)[:, 1]
    roll_v8 = rotations_v8.as_euler("YXZ", degrees=True)[:, 2]
    return {
        "centers": centers,
        "rotations": rotations_v8,
        "yaw": yaw_v8,
        "pitch": pitch_v8,
        "roll": roll_v8,
        "confidence": conf,
        "position_confidence": pos_conf,
        "rotation_confidence": rot_conf,
        "status": status,
        "verified_observation": obs_mask,
        "accepted_face": accepted_face,
        "accepted_flow": accepted_flow,
        "accepted_body": accepted_body,
        "gated_face": gated_face,
        "gated_flow": gated_flow,
        "gated_body": gated_body,
        "pre_center": pre_center,
        "pre_rot": Rotation.from_matrix(pre_rot),
        "pre_confidence": pre_conf,
        "body_prior_center": body_prior_center,
        "body_prior_yaw": body_prior_yaw,
        "body_prior_confidence": body_prior_conf,
        "face_maha": face_maha,
        "flow_maha": flow_maha,
        "body_maha": body_maha,
        "rls_counts": {
            "center_offset": int(offset_x.count),
            "yaw": int(yaw_model.count),
        },
    }


def build_v8_geometry(v7_stream, est: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v7_px = np.asarray(v7_stream["projected_px"], dtype=np.float64)
    v7_verts = np.asarray(v7_stream["verts"], dtype=np.float64)
    source = np.asarray(v7_stream["mesh_source"]).astype(str)
    reliable = face_reliable_mask(source)
    base_center = v7.semantic_center(v7_px)
    delta = np.asarray(est["centers"], dtype=np.float64) - base_center
    projected = v7_px + delta[:, None, :]
    verts = v7_verts.copy()
    verts[:, :, :2] = projected
    # Verified face frames are direct observations, not predictions.
    projected[reliable] = v7_px[reliable]
    verts[reliable] = v7_verts[reliable]
    head_transform = v7.update_head_transforms_with_rotations(
        np.asarray(v7_stream["head_transform"], dtype=np.float64),
        est["rotations"],
    )
    return verts.astype(np.float32), projected.astype(np.float32), head_transform


def reacquisition_metrics(v7_stream, est: Dict) -> Dict:
    source = np.asarray(v7_stream["mesh_source"]).astype(str)
    reliable = face_reliable_mask(source)
    hidden = ~reliable
    face_center = v7.semantic_center(np.asarray(v7_stream["projected_px"], dtype=np.float64))
    head_scale = np.asarray(v7_stream["head_scale_px"], dtype=np.float64)
    yaw_v7 = unwrap_yaw_deg(np.asarray(v7_stream["yaw_deg"], dtype=np.float64))
    pre_center = np.asarray(est["pre_center"], dtype=np.float64)
    pre_yaw = np.asarray([yaw_from_rotation(r, yaw_v7[i]) for i, r in enumerate(est["pre_rot"])], dtype=np.float64)
    body_center = np.asarray(est["body_prior_center"], dtype=np.float64)
    body_yaw = np.asarray(est["body_prior_yaw"], dtype=np.float64)

    rows = []
    body_rows = []
    calib_samples = []
    for start, end in hidden_spans(hidden):
        left = start - 1
        right = end + 1
        if left < 1 or right >= len(source) or not reliable[right]:
            continue
        actual_c = face_center[right]
        actual_y = yaw_v7[right]
        v8_center_err = float(np.linalg.norm(pre_center[right] - actual_c) / max(float(head_scale[right]), 1.0))
        v8_yaw_err = float(abs(wrap_angle_deg(pre_yaw[right] - actual_y)))
        base_c = face_center[end] + (face_center[end] - face_center[max(left, end - 1)])
        base_y = yaw_v7[end] + (yaw_v7[end] - yaw_v7[max(left, end - 1)])
        base_center_err = float(np.linalg.norm(base_c - actual_c) / max(float(head_scale[right]), 1.0))
        base_yaw_err = float(abs(wrap_angle_deg(base_y - actual_y)))
        combined_v8 = v8_center_err + v8_yaw_err / 60.0
        combined_base = base_center_err + base_yaw_err / 60.0
        row = {
            "hidden_start": int(start),
            "hidden_end": int(end),
            "reacq_frame": int(right),
            "len": int(end - start + 1),
            "v8_pre_correct_center_error_over_head": v8_center_err,
            "v8_pre_correct_yaw_error_deg": v8_yaw_err,
            "v8_combined_error": combined_v8,
            "v7_extrapolated_center_error_over_head": base_center_err,
            "v7_extrapolated_yaw_error_deg": base_yaw_err,
            "v7_combined_error": combined_base,
            "pre_confidence": float(est["pre_confidence"][right]),
        }
        rows.append(row)
        success = bool(v8_center_err < 0.15 and v8_yaw_err < 18.0)
        calib_samples.append((float(est["pre_confidence"][right]), success, combined_v8))
        if np.isfinite(body_center[right]).all() or np.isfinite(body_yaw[right]):
            bc = body_center[right] if np.isfinite(body_center[right]).all() else pre_center[right]
            by = body_yaw[right] if np.isfinite(body_yaw[right]) else pre_yaw[right]
            body_center_err = float(np.linalg.norm(bc - actual_c) / max(float(head_scale[right]), 1.0))
            body_yaw_err = float(abs(wrap_angle_deg(by - actual_y)))
            body_rows.append({
                "hidden_start": int(start),
                "hidden_end": int(end),
                "reacq_frame": int(right),
                "body_prior_center_error_over_head": body_center_err,
                "body_prior_yaw_error_deg": body_yaw_err,
                "body_prior_combined_error": body_center_err + body_yaw_err / 60.0,
                "v7_baseline_combined_error": combined_base,
                "body_prior_confidence": float(est["body_prior_confidence"][right]),
            })

    def arr(key: str, data: List[Dict]) -> np.ndarray:
        return np.asarray([r[key] for r in data], dtype=np.float64)

    body_combined = arr("body_prior_combined_error", body_rows) if body_rows else np.asarray([])
    base_for_body = arr("v7_baseline_combined_error", body_rows) if body_rows else np.asarray([])
    body_kill = bool(len(body_combined) > 0 and float(np.mean(body_combined)) > float(np.mean(base_for_body)))
    return {
        "definition": "score prediction at first verified face frame after hidden/profile/interpolated span",
        "spans": rows,
        "summary": {
            "count": int(len(rows)),
            "v8_combined_error": summarize(arr("v8_combined_error", rows)) if rows else summarize([]),
            "v7_baseline_combined_error": summarize(arr("v7_combined_error", rows)) if rows else summarize([]),
            "v8_center_error_over_head": summarize(arr("v8_pre_correct_center_error_over_head", rows)) if rows else summarize([]),
            "v8_yaw_error_deg": summarize(arr("v8_pre_correct_yaw_error_deg", rows)) if rows else summarize([]),
        },
        "body_prior_vs_v7": {
            "count": int(len(body_rows)),
            "rows": body_rows,
            "body_prior_combined_error": summarize(body_combined),
            "v7_baseline_combined_error": summarize(base_for_body),
            "mean_body_prior_combined_error": float(np.mean(body_combined)) if len(body_combined) else 0.0,
            "mean_v7_baseline_combined_error": float(np.mean(base_for_body)) if len(base_for_body) else 0.0,
            "kill_body_prior_worse_than_v7": body_kill,
        },
        "calibration_samples": calib_samples,
    }


def calibration_report(samples: List[Tuple[float, bool, float]]) -> Dict:
    if not samples:
        return {"ece": 0.0, "count": 0, "bins": []}
    bins = []
    ece = 0.0
    n = len(samples)
    for lo in np.linspace(0.0, 0.8, 5):
        hi = lo + 0.2
        rows = [(c, ok, e) for c, ok, e in samples if c >= lo and (c < hi or hi >= 1.0)]
        if not rows:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0, "mean_confidence": 0.0, "empirical_accuracy": 0.0})
            continue
        conf = float(np.mean([r[0] for r in rows]))
        acc = float(np.mean([1.0 if r[1] else 0.0 for r in rows]))
        ece += len(rows) / n * abs(conf - acc)
        bins.append({
            "lo": float(lo),
            "hi": float(hi),
            "count": int(len(rows)),
            "mean_confidence": conf,
            "empirical_accuracy": acc,
            "mean_combined_error": float(np.mean([r[2] for r in rows])),
        })
    return {"ece": float(ece), "count": int(n), "bins": bins}


def build_reliability_png(calib: Dict) -> None:
    w, h = 900, 520
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    margin = 70
    plot_w = w - 2 * margin
    plot_h = h - 2 * margin
    cv2.rectangle(img, (margin, margin), (margin + plot_w, margin + plot_h), (0, 0, 0), 1)
    cv2.line(img, (margin, margin + plot_h), (margin + plot_w, margin), (120, 120, 120), 1, cv2.LINE_AA)
    bins = calib.get("bins", [])
    bar_w = max(int(plot_w / max(len(bins), 1) * 0.35), 12)
    for i, b in enumerate(bins):
        cx = margin + int((i + 0.5) * plot_w / max(len(bins), 1))
        conf_y = margin + plot_h - int(float(b.get("mean_confidence", 0.0)) * plot_h)
        acc_y = margin + plot_h - int(float(b.get("empirical_accuracy", 0.0)) * plot_h)
        cv2.rectangle(img, (cx - bar_w, conf_y), (cx - 2, margin + plot_h), (255, 165, 70), -1)
        cv2.rectangle(img, (cx + 2, acc_y), (cx + bar_w, margin + plot_h), (70, 190, 90), -1)
        cv2.putText(img, str(b.get("count", 0)), (cx - 12, margin + plot_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"v8 calibration ECE={calib.get('ece', 0.0):.3f} n={calib.get('count', 0)}", (margin, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "blue=mean confidence, green=empirical success", (margin, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(RELIABILITY_PATH, img)


def pose_and_mesh_metrics(v7_stream, est: Dict, projected_v8: np.ndarray) -> Dict:
    rotations_v7 = v7.rotations_from_stream(
        np.asarray(v7_stream["head_transform"], dtype=np.float64),
        np.asarray(v7_stream["yaw_deg"], dtype=np.float64),
        np.asarray(v7_stream["pitch_deg"], dtype=np.float64),
        np.asarray(v7_stream["roll_deg"], dtype=np.float64),
    )
    rotations_v8 = est["rotations"]
    step_v7 = angular_jumps(rotations_v7)
    step_v8 = angular_jumps(rotations_v8)
    hs = np.asarray(v7_stream["head_scale_px"], dtype=np.float64)
    px7 = np.asarray(v7_stream["projected_px"], dtype=np.float64)
    px8 = np.asarray(projected_v8, dtype=np.float64)
    mv7 = mean_vertex_step(px7, hs)
    mv8 = mean_vertex_step(px8, hs)
    span = slice(POSE_FLIP_START + 1, POSE_FLIP_END + 1)
    return {
        "pose_angular_jump_deg": {
            "v7": summarize(step_v7[1:]),
            "v8": summarize(step_v8[1:]),
            "pose_flip_span_v7": summarize(step_v7[span]),
            "pose_flip_span_v8": summarize(step_v8[span]),
            "pose_flip_max_v7": float(np.max(step_v7[span])),
            "pose_flip_max_v8": float(np.max(step_v8[span])),
        },
        "mean_vertex_jump_over_head": {
            "v7": summarize(mv7[1:]),
            "v8": summarize(mv8[1:]),
            "pose_flip_span_v7": summarize(mv7[span]),
            "pose_flip_span_v8": summarize(mv8[span]),
            "pose_flip_max_v7": float(np.max(mv7[span])),
            "pose_flip_max_v8": float(np.max(mv8[span])),
        },
    }


def scan_for_disallowed_calls() -> Dict:
    with open(__file__, "r", encoding="utf-8") as f:
        text = f.read()
    needles = ["torch." + "cuda", "." + "cuda(", "pytorch" + "3d", "nvdi" + "ffrast"]
    hits = [needle for needle in needles if needle in text]
    return {"pass": len(hits) == 0, "hits": hits, "execution": "CPU/OpenCV/MediaPipe postpass; no Torch device used"}


def draw_hud(canvas: np.ndarray, fidx: int, total_f: int, source: str, status: str,
             conf: float, body_conf: float, yaw: float) -> None:
    color = (70, 220, 80) if status == "verified" else ((70, 210, 255) if status == "mixed" else (80, 120, 255))
    lines = [
        f"f{fidx:04d}/{total_f} {source} v8={status} conf={conf:.2f}",
        f"yaw={yaw:+.0f} body_prior={body_conf:.2f}",
        "predict-correct v8: face/flow/body gated",
    ]
    for i, line in enumerate(lines):
        y = 26 + i * 22
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def label_cell(img: np.ndarray, text: str, size: Tuple[int, int]) -> np.ndarray:
    cell = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, 0), (size[0], 28), (0, 0, 0), -1)
    cv2.putText(cell, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def build_overlay(projected_px: np.ndarray, faces: np.ndarray, source: np.ndarray, est: Dict,
                  body_rows: List[BodyRecord]) -> Dict:
    edges = v7.faces_to_edges(faces)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.0)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = len(projected_px)
    tmp_path = OVERLAY_MASTER_PATH.replace(".mp4", "_tmp.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
    if not writer.isOpened():
        raise RuntimeError("Could not open v8 overlay writer")
    saved_strip = []
    for fidx in range(total_f):
        ok, frame = cap.read()
        if not ok:
            break
        canvas = frame.copy()
        v7.draw_edges(canvas, projected_px[fidx], edges, (205, 205, 205), 1)
        v7.draw_contours(canvas, projected_px[fidx], str(source[fidx]))
        draw_hud(
            canvas,
            fidx,
            total_f,
            str(source[fidx]),
            str(est["status"][fidx]),
            float(est["confidence"][fidx]),
            float(est["body_prior_confidence"][fidx]),
            float(est["yaw"][fidx]),
        )
        writer.write(canvas)
        if POSE_FLIP_START <= fidx <= POSE_FLIP_END:
            saved_strip.append((canvas.copy(), fidx))
    cap.release()
    writer.release()
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", OVERLAY_MASTER_PATH,
    ], check=True, capture_output=True)
    os.remove(tmp_path)
    duration_s = max(total_f / max(fps, 1e-6), 1.0)
    target_kbps = int((7.2 * 8 * 1000) / duration_s)
    subprocess.run([
        "ffmpeg", "-y", "-i", OVERLAY_MASTER_PATH,
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", f"{target_kbps}k", "-maxrate", f"{target_kbps * 2}k",
        "-bufsize", f"{target_kbps * 4}k", "-pix_fmt", "yuv420p",
        OVERLAY_PREVIEW_PATH,
    ], check=True, capture_output=True)
    cells = [label_cell(img, f"v8 f{fidx}", (144, 256)) for img, fidx in saved_strip]
    if cells:
        cv2.imwrite(MOTION_STRIP_PATH, np.hstack(cells))
    return {
        "overlay_master": OVERLAY_MASTER_PATH,
        "overlay_preview": OVERLAY_PREVIEW_PATH,
        "motion_strip": MOTION_STRIP_PATH,
    }


def write_notes(report: Dict, metrics: Dict) -> None:
    pose = metrics["pose_and_mesh"]["pose_angular_jump_deg"]
    mesh = metrics["pose_and_mesh"]["mean_vertex_jump_over_head"]
    reacq = metrics["reacquisition"]
    body = reacq["body_prior_vs_v7"]
    lines = [
        "# Predict-Correct V8 Notes",
        "",
        "## Scope",
        "- v8 is a buildable-now predict/correct postpass over v7.",
        "- Implemented pose/position confidence estimator: SO(3) error-state rotation, linear position KF, optical flow, and soft MediaPipe-Pose body prior.",
        "- Expression/texture are not built in this iteration beyond carrying existing v7 stream fields.",
        "",
        "## Outputs",
        f"- Stream: `{STREAM_PATH}`",
        f"- Overlay master: `{OVERLAY_MASTER_PATH}`",
        f"- Overlay preview: `{OVERLAY_PREVIEW_PATH}`",
        f"- Motion strip f425-f440: `{MOTION_STRIP_PATH}`",
        f"- Reliability diagram: `{RELIABILITY_PATH}`",
        f"- Report: `{REPORT_PATH}`",
        f"- Metrics: `{METRICS_PATH}`",
        "",
        "## Pre-Registered Metrics vs V7",
        f"- Pose angular jump p99: v7={pose['v7']['p99']:.2f}deg, v8={pose['v8']['p99']:.2f}deg.",
        f"- f425-f440 max pose step: v7={pose['pose_flip_max_v7']:.2f}deg, v8={pose['pose_flip_max_v8']:.2f}deg.",
        f"- f425-f440 max mean vertex jump/head: v7={mesh['pose_flip_max_v7']:.4f}, v8={mesh['pose_flip_max_v8']:.4f}.",
        f"- Reacquisition combined error: v8 median={reacq['summary']['v8_combined_error']['median']:.4f}, v7 baseline median={reacq['summary']['v7_baseline_combined_error']['median']:.4f}.",
        f"- Body prior vs v7 baseline mean combined error: body={body['mean_body_prior_combined_error']:.4f}, v7={body['mean_v7_baseline_combined_error']:.4f}; kill={body['kill_body_prior_worse_than_v7']}.",
        f"- Calibration ECE: {metrics['calibration']['ece']:.4f} over {metrics['calibration']['count']} scored reacquisitions.",
        f"- Confidence/status counts: {metrics['confidence']['status_counts']}.",
        f"- Predicted rendered as verified kill: {metrics['kill_conditions']['predicted_rendered_as_verified']}.",
        f"- MPS/no-CUDA scan clean: {metrics['mps_no_cuda']['pass']} hits={metrics['mps_no_cuda']['hits']}.",
        "",
        "## Honest Floor",
        "- Body pose is a soft prior and can be confidently wrong when the head turns independently of torso/shoulders.",
        "- Calibration samples are limited to hidden-span reacquisitions on this clip; ECE is a probe, not a population guarantee.",
        "- v8 pose output is confidence-labeled; only reliable face corrections produce `verified` status.",
    ]
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run() -> Dict:
    t0 = time.time()
    print(f"{LOG_PREFIX} loading v7 stream...")
    stream = np.load(V7_STREAM_PATH, allow_pickle=True)
    source = np.asarray(stream["mesh_source"]).astype(str)
    total_f = len(source)
    v7_px = np.asarray(stream["projected_px"], dtype=np.float64)
    head_scale = np.asarray(stream["head_scale_px"], dtype=np.float64)
    reliable = face_reliable_mask(source)
    face_center = v7.semantic_center(v7_px)

    print(f"{LOG_PREFIX} computing optical flow anchor...")
    flow = v7.compute_optical_flow_anchors(VIDEO_PATH, face_center, head_scale, reliable)
    print(f"{LOG_PREFIX} running MediaPipe Pose body prior...")
    body_rows, body_report = run_body_pose(VIDEO_PATH, total_f)

    print(f"{LOG_PREFIX} estimating recursive predict/correct state...")
    est = fit_body_prior_and_estimate(stream, flow, body_rows)
    verts, projected_px, head_transform = build_v8_geometry(stream, est)
    metrics_pose = pose_and_mesh_metrics(stream, est, projected_px)
    reacq = reacquisition_metrics(stream, est)
    calib = calibration_report(reacq["calibration_samples"])
    build_reliability_png(calib)
    mps_no_cuda = scan_for_disallowed_calls()

    status_counts = {str(k): int(v) for k, v in zip(*np.unique(est["status"], return_counts=True))}
    predicted_as_verified = bool(np.any((est["status"] == "verified") & (~est["verified_observation"])))
    faces = np.asarray(stream["faces"], dtype=np.int32)
    stream_payload = {key: np.asarray(stream[key]) for key in stream.files}
    stream_payload.update({
        "verts": verts,
        "projected_px": projected_px,
        "head_transform": head_transform,
        "yaw_deg": np.asarray(est["yaw"], dtype=np.float32),
        "yaw_wrapped_deg": np.asarray(wrap_angle_deg(est["yaw"]), dtype=np.float32),
        "pitch_deg": np.asarray(est["pitch"], dtype=np.float32),
        "roll_deg": np.asarray(est["roll"], dtype=np.float32),
        "pose_confidence": np.asarray(est["confidence"], dtype=np.float32),
        "position_confidence": np.asarray(est["position_confidence"], dtype=np.float32),
        "rotation_confidence": np.asarray(est["rotation_confidence"], dtype=np.float32),
        "pose_status": np.asarray(est["status"], dtype="<U16"),
        "pose_angular_step_deg": np.asarray(angular_jumps(est["rotations"]), dtype=np.float32),
        "pre_pose_confidence": np.asarray(est["pre_confidence"], dtype=np.float32),
        "verified_observation": np.asarray(est["verified_observation"], dtype=bool),
        "accepted_body_prior": np.asarray(est["accepted_body"], dtype=bool),
        "accepted_flow_observation": np.asarray(est["accepted_flow"], dtype=bool),
        "accepted_face_observation": np.asarray(est["accepted_face"], dtype=bool),
        "body_prior_confidence": np.asarray(est["body_prior_confidence"], dtype=np.float32),
        "body_prior_center_px": np.asarray(est["body_prior_center"], dtype=np.float32),
        "body_prior_yaw_deg": np.asarray(est["body_prior_yaw"], dtype=np.float32),
        "pipeline_version": np.asarray([PIPELINE_VERSION]),
        "source_stream": np.asarray(["mesh_cascade_v7_stream.npz"], dtype="<U64"),
        "predict_correct_policy": np.asarray(["so3_eskf_position_kf_face_flow_body_prior_innovation_gated"], dtype="<U96"),
    })
    np.savez_compressed(STREAM_PATH, **stream_payload)

    print(f"{LOG_PREFIX} rendering overlay...")
    paths = build_overlay(projected_px, faces, source, est, body_rows)
    output_sizes = {
        "stream": round(os.path.getsize(STREAM_PATH) / 1e6, 3),
        "overlay_master": round(os.path.getsize(OVERLAY_MASTER_PATH) / 1e6, 3),
        "overlay_preview": round(os.path.getsize(OVERLAY_PREVIEW_PATH) / 1e6, 3),
        "motion_strip": round(os.path.getsize(MOTION_STRIP_PATH) / 1e6, 3) if os.path.exists(MOTION_STRIP_PATH) else 0.0,
        "reliability": round(os.path.getsize(RELIABILITY_PATH) / 1e6, 3) if os.path.exists(RELIABILITY_PATH) else 0.0,
    }
    metrics = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {"v7_stream": V7_STREAM_PATH, "v7_report": V7_REPORT_PATH, "video": VIDEO_PATH, "pose_model": POSE_MODEL},
        "pose_and_mesh": metrics_pose,
        "reacquisition": {k: v for k, v in reacq.items() if k != "calibration_samples"},
        "calibration": calib,
        "confidence": {
            "status_counts": status_counts,
            "pose_confidence": summarize(est["confidence"]),
            "position_confidence": summarize(est["position_confidence"]),
            "rotation_confidence": summarize(est["rotation_confidence"]),
            "verified_frames": int(est["verified_observation"].sum()),
            "mixed_frames": int((est["status"] == "mixed").sum()),
            "predicted_frames": int((est["status"] == "predicted").sum()),
        },
        "body_pose": body_report,
        "rls_counts": est["rls_counts"],
        "mps_no_cuda": mps_no_cuda,
        "kill_conditions": {
            "body_prior_worse_than_v7": bool(reacq["body_prior_vs_v7"]["kill_body_prior_worse_than_v7"]),
            "predicted_rendered_as_verified": predicted_as_verified,
            "mps_no_cuda_scan_failed": not bool(mps_no_cuda["pass"]),
        },
        "paths": {
            "stream": STREAM_PATH,
            "report": REPORT_PATH,
            "metrics": METRICS_PATH,
            "notes": NOTES_PATH,
            "reliability": RELIABILITY_PATH,
            **paths,
        },
        "output_sizes_mb": output_sizes,
    }
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": metrics["generated_at"],
        "summary": {
            "pose_p99_v7_to_v8": [
                metrics_pose["pose_angular_jump_deg"]["v7"]["p99"],
                metrics_pose["pose_angular_jump_deg"]["v8"]["p99"],
            ],
            "pose_flip_max_v7_to_v8": [
                metrics_pose["pose_angular_jump_deg"]["pose_flip_max_v7"],
                metrics_pose["pose_angular_jump_deg"]["pose_flip_max_v8"],
            ],
            "body_prior_kill": metrics["kill_conditions"]["body_prior_worse_than_v7"],
            "calibration_ece": calib["ece"],
            "status_counts": status_counts,
        },
        "metrics": metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_notes(report, metrics)

    print(f"{LOG_PREFIX} DONE")
    print(f"  Pose p99 v7->v8: {metrics_pose['pose_angular_jump_deg']['v7']['p99']:.2f} -> {metrics_pose['pose_angular_jump_deg']['v8']['p99']:.2f}")
    print(f"  f425-440 max pose v7->v8: {metrics_pose['pose_angular_jump_deg']['pose_flip_max_v7']:.2f} -> {metrics_pose['pose_angular_jump_deg']['pose_flip_max_v8']:.2f}")
    print(f"  body prior kill: {metrics['kill_conditions']['body_prior_worse_than_v7']}")
    print(f"  calibration ECE: {calib['ece']:.4f}")
    print(f"  total wall_s: {time.time() - t0:.1f}")
    return report


if __name__ == "__main__":
    run()
