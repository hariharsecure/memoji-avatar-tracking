#!/usr/bin/env python3
"""
avatar_overlay_pipeline_v16.py — Rendering placement + size bug-fix.

BUG FIXED vs v15:
  Visual inspection of avatar_overlay_v15_montage.png confirmed two bugs:
  - Frontal (f50) and close-up (f828, f844): Memoji too small, sitting at chin/neck.
  - Profile (f300, f437, f548, f694, f758): Memoji large, correctly on head.

  ROOT CAUSE 1 — WRONG PLACEMENT DIRECTION:
    v15 used EAR_TO_FACE_FRAC = +0.55 (shift DOWN from ear-midpoint by 55% of sc_px).
    The calibration reference was YOLO y1 + 0.80*bbox_h = CHIN level, not face center.
    Result: Memoji was pulled to the chin in frontal/close-up frames.

    CORRECT TARGET: eye-midpoint (MediaPipe landmarks 159/386 center).
    Measured across 32 MEDIAPIPE frames with sc_px >= 50:
      (eye_mid_y − rig_ear_cy) / sc_px: mean=-0.141, median=-0.143, SD=0.167
    The eye-midpoint is ABOVE the ear-midpoint (negative fraction = shift UP).
    FIX: EAR_TO_FACE_FRAC = -0.143 (shift upward by 0.143 * sc_px).

  ROOT CAUSE 2 — INVERTED SIZE FORMULA:
    v15 used: avatar_scale = 9.0 * (100 / sc_px)   ← INVERSELY proportional.
    Larger head (sc_px=263 close-up) → avatar_scale=3.4 (tiny).
    Smaller head (sc_px=69 profile) → avatar_scale=13.0 (huge).

    CORRECT: avatar_scale should be DIRECTLY proportional to sc_px.
    Physics: rendered_diameter = 2 * avatar_scale * focal / |z_ref|
    To match avatar diameter to face width (sc_px):
      avatar_scale = sc_px * |z_ref| / (2 * focal) * coverage_factor
    With z_ref=-88.8, focal=700:
      base_k = 88.8 / 1400 = 0.0634
    Calibrated against YOLO bbox_w (face coverage factor ~1.185x ear-span):
      SCALE_K_MEDIAPIPE = 0.0752  (32 frontal/close-up frames, YOLO bbox_w/sc median=1.185)

    REP360 mode: pose ear-span (sc_px) collapses in profile (one ear hidden → shoulder-span
    fallback ~60-70px). YOLO bbox_w / sc_px ratio is 1.35x larger than MEDIAPIPE.
      SCALE_K_REP360 = 0.0752 * 1.35 = 0.1016  (91 profile frames, median ratio=1.35)

  RETAINED from v15: ear-midpoint anchor, Kalman smoother, 100% coverage.

Python: python3
"""
from __future__ import annotations
import json, math, os, struct, subprocess, time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrender
import trimesh

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH    = "input_clip.mov"
RIG_NPZ_V15   = "./memoji_rig_stream_v13.npz"
GLB_PATH      = "assets/emoji_head.glb"
OUT_DIR       = "."
os.makedirs(OUT_DIR, exist_ok=True)

FOCAL_LEN     = 700.0

# ── PLACEMENT FIX (v16): EAR→EYE-MIDPOINT OFFSET ──────────────────────────
# The v15 rig anchor is the ear-midpoint (lm234+lm454 midpoint).
# The Memoji GLB is centered at face-level (between eyes / nose-bridge).
# Eye-midpoint sits ABOVE the ear-midpoint in image-space (y positive = down).
#
# Calibration (2026-06-14): 32 MEDIAPIPE frames, sc_px >= 50, eye_mid_y from lm159/386:
#   (eye_mid_y − rig_ear_cy) / sc_px: mean=-0.141, median=-0.143, SD=0.167
#   filtered (|frac|<0.5): n=28, mean=-0.141, median=-0.143
# Using -0.143 (shift upward).
# Applied as: cy_face = cy_ear + EAR_TO_FACE_FRAC * sc_px  (negative = upward)
EAR_TO_FACE_FRAC = -0.143

# ── SIZE FIX (v16): DIRECT-PROPORTIONAL SCALE ──────────────────────────────
# Correct formula: avatar_scale = SCALE_K * sc_px (direct, not inverse)
# MEDIAPIPE (frontal/close-up): sc_px = ear-span = reliable face width indicator
#   YOLO bbox_w / sc_px median = 1.185 across 32 frames
#   SCALE_K = 0.0634 * 1.185 = 0.0752
# REP360 (profile/back): sc_px = pose ear-span (collapses in profile)
#   YOLO bbox_w / sc_px median = 1.35 across 91 frames vs 1.185 for MP
#   Ratio correction: 1.35 / 1.185 * 0.0752 = 0.0856 ≈ 0.0856
#   But: calibrate directly: SCALE_K * 1.35 = 0.0752 * 1.35 = 0.1015
# HOLD: use REP360 scale (pose-based)
SCALE_K_MEDIAPIPE = 0.0752
SCALE_K_REP360    = 0.1016   # = 0.0752 * 1.35 (profile ear-span correction)
SCALE_K_HOLD      = 0.1016   # same as REP360 (hold = last pose)

SCALE_MIN = 3.0
SCALE_MAX = 25.0


# ─────────────────────────────────────────────────────────────────────────────
# GLB loader (identical to v13/v15)
# ─────────────────────────────────────────────────────────────────────────────
class EmojiGLB:
    def __init__(self, path: str):
        raw = open(path, 'rb').read()
        c0l = struct.unpack('<I', raw[12:16])[0]
        self.jd  = json.loads(raw[20:20+c0l])
        bs = (20 + c0l + 3) & ~3
        self.bin = raw[bs+8:]
        self.accs = self.jd['accessors']
        self.bvs  = self.jd['bufferViews']
        self.parts = self._load_parts()

    def _racc(self, ai: int) -> np.ndarray:
        acc = self.accs[ai]
        count = acc['count']
        nc  = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[acc['type']]
        dt  = {5126:np.float32,5121:np.uint8,
               5123:np.uint16,5125:np.uint32}[acc['componentType']]
        bv  = acc.get('bufferView')
        if bv is not None:
            bvd = self.bvs[bv]
            off = bvd.get('byteOffset',0) + acc.get('byteOffset',0)
            a   = np.frombuffer(self.bin[off:off+count*nc*np.dtype(dt).itemsize], dtype=dt)
            return a.reshape(count, nc) if nc > 1 else a.copy()
        out = np.zeros((count, nc) if nc > 1 else count, dtype=np.float32)
        sp  = acc.get('sparse', {})
        if sp:
            n  = sp['count']
            ic = {5121:np.uint8,5123:np.uint16,5125:np.uint32}[
                sp['indices']['componentType']]
            ibv = self.bvs[sp['indices']['bufferView']]
            io  = ibv.get('byteOffset',0) + sp['indices'].get('byteOffset',0)
            idx = np.frombuffer(self.bin[io:io+n*np.dtype(ic).itemsize], dtype=ic)
            vbv = self.bvs[sp['values']['bufferView']]
            vo  = vbv.get('byteOffset',0) + sp['values'].get('byteOffset',0)
            va  = np.frombuffer(self.bin[vo:vo+n*nc*4], dtype=np.float32)
            if nc > 1: va = va.reshape(n, nc)
            out[idx] = va
        return out

    def _load_parts(self) -> List[Dict]:
        parts = []
        for ni, node in enumerate(self.jd['nodes']):
            mesh_i = node.get('mesh')
            if mesh_i is None:
                continue
            m     = self.jd['meshes'][mesh_i]
            prim  = m['primitives'][0]
            base  = self._racc(prim['attributes']['POSITION'])
            faces = self._racc(prim['indices']).astype(np.int32).ravel().reshape(-1, 3)
            tnames = m.get('extras', {}).get('targetNames', [])
            deltas = [self._racc(prim['targets'][ti]['POSITION'])
                      for ti in range(len(prim.get('targets', [])))]
            s = np.array(node.get('scale',       [1, 1, 1]), dtype=np.float32)
            t = np.array(node.get('translation', [0, 0, 0]), dtype=np.float32)
            base_b  = base * s[None, :] + t[None, :]
            deltas_b = [d * s[None, :] for d in deltas]
            base_c = base_b.copy(); base_c[:, 1] *= -1; base_c[:, 2] *= -1
            deltas_c = []
            for d in deltas_b:
                dc = d.copy(); dc[:, 1] *= -1; dc[:, 2] *= -1
                deltas_c.append(dc)
            mat_i = prim.get('material')
            color = np.array([0.9, 0.80, 0.65, 1.0], dtype=np.float32)
            if mat_i is not None:
                pbr = self.jd['materials'][mat_i].get('pbrMetallicRoughness', {})
                bc  = pbr.get('baseColorFactor')
                if bc: color = np.array(bc, dtype=np.float32)
            parts.append({
                'name':   node.get('name', f'node{ni}'),
                'base':   base_c,
                'faces':  faces,
                'tnames': tnames,
                'deltas': deltas_c,
                'color':  color,
            })
        return parts

    def morphed_verts(self, part: Dict, weights: Dict[str, float]) -> np.ndarray:
        pos = part['base'].copy()
        for ti, tn in enumerate(part['tnames']):
            w = weights.get(tn, 0.0)
            if abs(w) > 1e-5 and ti < len(part['deltas']):
                pos = pos + w * part['deltas'][ti]
        return pos


# ─────────────────────────────────────────────────────────────────────────────
# Transform builder (identical to v13/v15)
# ─────────────────────────────────────────────────────────────────────────────
def extract_unit_rotation(T: np.ndarray, mode: str) -> np.ndarray:
    T = T.astype(np.float64)
    if str(mode) == 'MEDIAPIPE':
        return T[:3, :3].copy()
    col_norm = np.linalg.norm(T[:3, 0])
    col_norm = max(col_norm, 1e-6)
    return T[:3, :3] / col_norm


def build_T_from_screen_pos(R_unit: np.ndarray,
                             cx_px: float, cy_px: float,
                             fw: int, fh: int,
                             focal: float, z_ref: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_unit
    T[0, 3] = (cx_px - fw / 2.0) * (-z_ref) / focal
    T[1, 3] = (fh - cy_px - fh / 2.0) * (-z_ref) / focal
    T[2, 3] = z_ref
    return T


def compute_avatar_scale(sc_px: float, mode: str) -> float:
    """
    v16 FIX: DIRECT-proportional scale, mode-specific coefficient.

    MEDIAPIPE (frontal/close-up): sc_px = ear-span = reliable face width proxy.
      avatar_scale = SCALE_K_MEDIAPIPE * sc_px
    REP360 (profile/back): pose ear-span collapses in profile → sc_px underestimates face.
      Empirical correction: YOLO bbox_w ~1.35x larger than sc_px in profile frames.
      avatar_scale = SCALE_K_REP360 * sc_px (= SCALE_K_MEDIAPIPE * 1.35 * sc_px)
    HOLD: same as REP360 (last pose held, back-of-head).
    """
    if str(mode) == 'MEDIAPIPE':
        k = SCALE_K_MEDIAPIPE
    else:
        k = SCALE_K_REP360  # covers REP360 and HOLD
    return float(np.clip(k * max(sc_px, 10.0), SCALE_MIN, SCALE_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# Renderer (identical to v13/v15)
# ─────────────────────────────────────────────────────────────────────────────
class AvatarRenderer:
    def __init__(self, fw: int, fh: int, focal: float = FOCAL_LEN):
        self.fw, self.fh = fw, fh
        self.focal = focal
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=[0.35, 0.35, 0.35],
        )
        cam = pyrender.IntrinsicsCamera(
            fx=focal, fy=focal, cx=fw/2.0, cy=fh/2.0,
            znear=0.5, zfar=10000.0,
        )
        self.scene.add(cam, pose=np.eye(4))
        for pos, col, intensity in [
            ([0, 0, 200],   [1.0, 1.0, 1.0], 3.5),
            ([-80, 60, 150], [0.7, 0.85, 1.0], 2.0),
            ([80, -20, 100], [1.0, 0.95, 0.8], 1.2),
        ]:
            dl = pyrender.DirectionalLight(color=col, intensity=intensity)
            lp = np.eye(4); lp[:3, 3] = pos
            self.scene.add(dl, pose=lp)
        self.renderer = pyrender.OffscreenRenderer(fw, fh)

    def render(self, glb: 'EmojiGLB',
               T_norm: np.ndarray,
               weights: Dict[str, float],
               scale: float) -> np.ndarray:
        S = np.eye(4, dtype=np.float64)
        S[0, 0] = S[1, 1] = S[2, 2] = scale
        T_node = T_norm @ S
        added = []
        for part in glb.parts:
            pos = glb.morphed_verts(part, weights)
            tm  = trimesh.Trimesh(vertices=pos, faces=part['faces'], process=False)
            tm.fix_normals()
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=part['color'].tolist(),
                metallicFactor=0.05, roughnessFactor=0.65, smooth=True,
            )
            pm   = pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True)
            added.append(self.scene.add(pm, pose=T_node))
        color, _ = self.renderer.render(
            self.scene,
            flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
        for node in added:
            self.scene.remove_node(node)
        return color

    def close(self):
        self.renderer.delete()


def composite_rgba_bgr(frame_bgr: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb   = rgba[:, :, :3].astype(np.float32)[:, :, ::-1]
    return np.clip(alpha * rgb + (1.0 - alpha) * frame_bgr.astype(np.float32),
                   0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print("[v16-overlay] Loading v15 rig stream...")
    rig          = np.load(RIG_NPZ_V15, allow_pickle=True)
    frames_arr   = rig['frame']
    modes_arr    = rig['mode']
    head_Ts      = rig['head_transform']
    blendshapes  = rig['blendshapes']
    arkit_names  = list(rig['arkit_names'])
    yaw_deg_arr  = rig['yaw_deg']
    head_center  = rig['head_center_px']    # [N,2] — v15 EAR-MIDPOINT anchors
    head_scale   = rig['head_scale_px']     # [N]
    anchor_src   = rig['anchor_source']     # [N]
    anchor_conf  = rig['anchor_confidence'] # [N]
    pipeline_ver = str(rig.get('pipeline_version', ['unknown'])[0])

    n_frames = int(len(frames_arr))
    print(f"  {n_frames} frames  pipeline_version={pipeline_ver}")
    # P2-B version assert: catch stale or wrong-version stream before rendering
    assert pipeline_ver == 'v15', (
        f"[v16-overlay] Expected pipeline_version='v15' in NPZ, got '{pipeline_ver}'. "
        "Run pipeline_memoji_rig_v15.py to regenerate the stream before rendering."
    )
    print(f"  EAR_TO_FACE_FRAC = {EAR_TO_FACE_FRAC}  (negative = shift UP to eye-midpoint)")
    print(f"  SCALE_K_MEDIAPIPE = {SCALE_K_MEDIAPIPE}  SCALE_K_REP360 = {SCALE_K_REP360}  (direct proportionality)")

    # z_ref from MP frames
    mp_mask  = modes_arr == 'MEDIAPIPE'
    mp_z_ref = float(np.median(head_Ts[mp_mask][:, 2, 3]))
    print(f"  MP z_ref = {mp_z_ref:.1f}")

    print("[v16-overlay] Loading GLB...")
    glb = EmojiGLB(GLB_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[v16-overlay] {fw}x{fh} @ {fps:.1f}fps")

    print("[v16-overlay] Init renderer...")
    av_renderer = AvatarRenderer(fw, fh, focal=FOCAL_LEN)

    raw_mp4     = f"{OUT_DIR}/avatar_v16_raw.mp4"
    master_mp4  = f"{OUT_DIR}/avatar_overlay_v15_master.mp4"    # overwrite v15
    preview_mp4 = f"{OUT_DIR}/avatar_overlay_v15_preview.mp4"   # overwrite v15
    montage_png = f"{OUT_DIR}/avatar_overlay_v15_montage.png"   # overwrite v15

    vw = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

    # Montage targets: frontal, profile, chin-up, extreme-yaw, AND close-ups f828/f844
    save_targets = {
        50:  "frontal_early",
        300: "frontal_mid",
        437: "reacq_f437",
        548: "extreme_yaw",
        694: "profile_f694",
        758: "profile_f758",
        828: "closeup_sit",
        844: "closeup_seated",
    }
    saved: Dict[str, np.ndarray] = {}

    color_map = {
        'MEDIAPIPE': (0, 220, 60),
        'REP360':    (30, 160, 255),
        'HOLD':      (220, 30, 220),
    }

    overlay_count  = 0
    t_start        = time.time()
    frame_positions: Dict[int, Tuple[float, float, float, float, float]] = {}
    # (cx_ear, cy_ear, cx_face, cy_face, avatar_scale)

    print("[v16-overlay] Processing frames...")
    for fidx in range(n_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        T_raw  = head_Ts[fidx]
        bs     = blendshapes[fidx]
        mode   = str(modes_arr[fidx])
        yaw    = float(yaw_deg_arr[fidx])
        asrc   = str(anchor_src[fidx])
        aconf  = float(anchor_conf[fidx])
        cx_ear = float(head_center[fidx, 0])
        cy_ear = float(head_center[fidx, 1])
        sc_px  = float(head_scale[fidx])

        # ── FIX 1: EAR→EYE-MIDPOINT OFFSET (v16) ────────────────────────────
        # EAR_TO_FACE_FRAC = -0.143 → shift UPWARD in image (negative = up)
        cy_offset = EAR_TO_FACE_FRAC * max(sc_px, 10.0)
        cx_face   = cx_ear
        cy_face   = float(np.clip(cy_ear + cy_offset, 0, fh - 1))

        # ── FIX 2: DIRECT-PROPORTIONAL SCALE (v16) ───────────────────────────
        avatar_scale = compute_avatar_scale(sc_px, mode)

        weights = {arkit_names[i]: float(bs[i]) for i in range(len(arkit_names))}
        frame_positions[fidx] = (cx_ear, cy_ear, cx_face, cy_face, avatar_scale)

        R_unit = extract_unit_rotation(T_raw, mode)
        T_norm = build_T_from_screen_pos(R_unit, cx_face, cy_face, fw, fh,
                                          FOCAL_LEN, mp_z_ref)

        rgba = av_renderer.render(glb, T_norm, weights, scale=avatar_scale)
        if int(rgba[:, :, 3].max()) > 10:
            overlay_count += 1
        out_frame = composite_rgba_bgr(frame_bgr, rgba)

        # OSD
        osd_col = color_map.get(mode, (200, 200, 200))
        jaw_open = float(weights.get('jawOpen', 0.0))
        eye_bl   = float(weights.get('eyeBlinkLeft', 0.0))
        smile_l  = float(weights.get('mouthSmileLeft', 0.0))
        cv2.putText(out_frame,
                    f"f{fidx:04d} [{mode}] yaw={yaw:.0f}° sc={sc_px:.0f}px",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, osd_col, 2)
        cv2.putText(out_frame,
                    f"anch={asrc[:12]} conf={aconf:.2f} scale={avatar_scale:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, osd_col, 1)

        # Ear-midpoint dot (yellow) + face-center dot (green)
        cv2.circle(out_frame, (int(cx_ear), int(cy_ear)), 5, (0, 220, 255), -1)
        cv2.circle(out_frame, (int(cx_face), int(cy_face)), 6, (0, 200, 0), -1)
        cv2.circle(out_frame, (int(cx_face), int(cy_face)), 8, (0, 0, 0), 1)

        def bar(img, x, val, col, lbl):
            bh = max(2, int(val * 70))
            cv2.rectangle(img, (x, fh-bh-5), (x+16, fh-5), col, -1)
            cv2.putText(img, lbl, (x-2, fh-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)
        bar(out_frame, fw-85, jaw_open, (0,255,255), "jaw")
        bar(out_frame, fw-60, eye_bl,   (255,180,0), "eyL")
        bar(out_frame, fw-35, smile_l,  (0,200,100), "sml")

        vw.write(out_frame)

        if fidx in save_targets:
            key = save_targets[fidx]
            saved[key] = out_frame.copy()
            cv2.imwrite(f"{OUT_DIR}/avatar_v16_proof_{key}_f{fidx:04d}.jpg", out_frame)

        if fidx % 50 == 0:
            e = time.time() - t_start
            fps_proc = (fidx+1) / max(e, 0.01)
            eta = (n_frames - fidx - 1) / max(fps_proc, 0.01)
            pct = 100 * overlay_count / max(fidx+1, 1)
            print(f"  f{fidx:04d}/{n_frames}  [{mode:<10}]  yaw={yaw:+5.0f}°  "
                  f"sc={sc_px:.0f}px  avsc={avatar_scale:.1f}  overlay={pct:.0f}%  "
                  f"{fps_proc:.1f}fps  ETA {eta:.0f}s")

    cap.release()
    vw.release()
    av_renderer.close()

    overlay_pct = 100.0 * overlay_count / n_frames

    # ── Montage ──────────────────────────────────────────────────────────────
    print("[v16-overlay] Building montage...")
    montage_targets = [
        (50,  "frontal_early",  "f050 FRONTAL"),
        (300, "frontal_mid",    "f300 PROFILE"),
        (437, "reacq_f437",     "f437 REACQ"),
        (548, "extreme_yaw",    "f548 EXTREME YAW"),
        (694, "profile_f694",   "f694 PROFILE"),
        (758, "profile_f758",   "f758 PROFILE"),
        (828, "closeup_sit",    "f828 CLOSE-UP"),
        (844, "closeup_seated", "f844 CLOSE-UP"),
    ]
    tw, th = 360, 640
    cols = 4; rows = 2
    montage = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    def load_frame(key: str, fname: str) -> np.ndarray:
        p = f"{OUT_DIR}/{fname}"
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                return cv2.resize(img, (tw, th))
        if key in saved:
            return cv2.resize(saved[key], (tw, th))
        return np.full((th, tw, 3), 30, dtype=np.uint8)

    for i, (fi, key, title) in enumerate(montage_targets):
        r = i // cols; c = i % cols
        fname = f"avatar_v16_proof_{key}_f{fi:04d}.jpg"
        thumb = load_frame(key, fname)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            cv2.putText(thumb, title, (8+dx, 26+dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 2)
        cv2.putText(thumb, title, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 60), 1)
        montage[r*th:(r+1)*th, c*tw:(c+1)*tw] = thumb

    cv2.imwrite(montage_png, montage)
    print(f"  Montage: {montage_png}")

    # ── Compress videos ───────────────────────────────────────────────────
    print("[v16-overlay] Compressing master (CRF 22)...")
    subprocess.run(["ffmpeg", "-y", "-i", raw_mp4,
                    "-vcodec", "libx264", "-crf", "22", "-preset", "fast",
                    "-movflags", "+faststart", master_mp4],
                   check=True, capture_output=True)

    print("[v16-overlay] Compressing preview version...")
    cap_check = cv2.VideoCapture(VIDEO_PATH)
    fw_v = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_v = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_check.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw_mp4,
                    "-vf", f"scale={fw_v//2}:{fh_v//2}",
                    "-vcodec", "libx264", "-crf", "30", "-preset", "fast",
                    "-movflags", "+faststart", preview_mp4],
                   check=True, capture_output=True)
    os.remove(raw_mp4)

    master_mb  = os.path.getsize(master_mp4)  / 1024**2
    preview_mb = os.path.getsize(preview_mp4) / 1024**2
    elapsed    = time.time() - t_start

    # ── Visual position report ────────────────────────────────────────────
    # For each montage frame: report where cy_face is relative to cy_ear and sc_px
    pos_report = []
    for fi, key, title in montage_targets:
        pos = frame_positions.get(fi)
        if pos:
            cx_ear, cy_ear, cx_face, cy_face, av_sc = pos
            sc = float(head_scale[fi])
            mode = str(modes_arr[fi])
            offset_frac = (cy_face - cy_ear) / max(sc, 1.0)
            pos_report.append({
                'frame': fi, 'label': title, 'mode': mode,
                'sc_px': round(sc, 1),
                'cy_ear': round(cy_ear, 1),
                'cy_face': round(cy_face, 1),
                'offset_px': round(cy_face - cy_ear, 1),
                'offset_frac': round(offset_frac, 3),
                'avatar_scale': round(av_sc, 2),
            })

    # ── Report ─────────────────────────────────────────────────────────────
    report = {
        "pipeline": "avatar_overlay_v16",
        "rig_version": pipeline_ver,
        "bug_fixes": {
            "placement": {
                "v15_frac": 0.55,
                "v16_frac": EAR_TO_FACE_FRAC,
                "description": (
                    "v15 calibrated against YOLO y1+0.80*bbox_h (chin level) → "
                    "Memoji was at chin in frontal/close-up. "
                    "v16 calibrated against eye-midpoint (MP lm159/386): "
                    "median frac=-0.143 from 28 frontal frames. "
                    "Eye-midpoint is ABOVE ear-midpoint → negative offset = shift up."
                ),
            },
            "scale": {
                "v15_formula": "9.0 * (100 / sc_px)  [INVERTED — large head → tiny avatar]",
                "v16_formula_mediapipe": f"SCALE_K_MEDIAPIPE * sc_px = {SCALE_K_MEDIAPIPE} * sc_px",
                "v16_formula_rep360": f"SCALE_K_REP360 * sc_px = {SCALE_K_REP360} * sc_px",
                "description": (
                    "v15 formula was inversely proportional: close-up (sc=263) → scale=3.4 (tiny), "
                    "profile (sc=69) → scale=13 (huge). "
                    "v16 direct proportionality. Mode-specific: REP360 sc_px collapses in profile "
                    "(pose ear-span) so multiplied by 1.35 (YOLO bbox_w/sc_px median ratio)."
                ),
            },
        },
        "overlay_coverage": {
            "frames_with_overlay": overlay_count,
            "total_frames": n_frames,
            "pct": overlay_pct,
        },
        "per_frame_position": pos_report,
        "outputs": {
            "rig_stream": RIG_NPZ_V15,
            "master":  {"path": master_mp4, "mb": round(master_mb, 2)},
            "preview": {"path": preview_mp4, "mb": round(preview_mb, 2)},
            "montage": montage_png,
        },
        "processing_time_s": round(elapsed, 1),
    }

    report_path = f"{OUT_DIR}/avatar_overlay_v15_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("AVATAR OVERLAY v16 — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"BUG FIX 1 — PLACEMENT: EAR_TO_FACE_FRAC {0.55} → {EAR_TO_FACE_FRAC} (eye not chin)")
    print(f"BUG FIX 2 — SIZE: 9.0*(100/sc) → SCALE_K*sc (direct, not inverse)")
    print(f"Overlay: {overlay_count}/{n_frames} = {overlay_pct:.1f}%")
    print(f"Master:  {master_mp4}  ({master_mb:.1f} MB)")
    print(f"Preview: {preview_mp4}  ({preview_mb:.1f} MB)")
    print(f"Montage: {montage_png}")
    print(f"\nPer-frame position report:")
    for d in pos_report:
        print(f"  f{d['frame']:04d} [{d['mode']:10s}] {d['label']:20s}  "
              f"cy_ear={d['cy_ear']:.0f} cy_face={d['cy_face']:.0f} "
              f"offset={d['offset_px']:+.0f}px ({d['offset_frac']:+.3f}*sc)  "
              f"avatar_scale={d['avatar_scale']:.1f}")
    print(f"{'='*70}")
    return report


if __name__ == '__main__':
    run()
