# Setup

This repo contains the pipeline code and design docs **only**. Third-party model
weights, assets, one vendored dependency, the input clip, and all intermediate
`.npz`/`.json`/video artifacts are **not** redistributed here (see `.gitignore`)
— you fetch the models and supply your own clip with the steps below.

The pipeline targets **Python 3.11** and runs on **Apple Silicon (MPS)** with no
CUDA dependency. It also runs on CPU; an NVIDIA GPU is not required.

> **Honesty note on reproducing the headline numbers.** The metrics quoted in
> `README.md` and the `notes_*.md` / `*_report.md` files were measured from a
> local run on one private monocular clip (847 frames, 720×1280, 29 fps). That
> clip and the per-frame `.npz` streams it produced are **not redistributed**.
> A clean clone can run the full pipeline end-to-end on *your own* clip and will
> reproduce the *behaviour* (per-frame coverage, the `verified`/`predicted`
> tagging, the overlay), but the exact published numbers are tied to that one
> source clip and are reported, not bit-for-bit reproducible from this repo
> alone. Reports are labelled `[MEASURED]` vs `[PUBLISHED]` accordingly.

---

## 1. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the core dependencies (PyTorch + torchvision,
MediaPipe, Ultralytics/YOLO, OpenCV, NumPy, SciPy, trimesh, pyrender,
onnxruntime, matplotlib, pyyaml, librosa).

## 2. System dependency: ffmpeg

Several pipelines (including the current head `pipeline_predict_correct_v8.py`,
the `pipeline_mesh_cascade_*` scripts, and the avatar-overlay renderers) shell
out to **`ffmpeg`** to transcode the raw overlay to a web-friendly H.264 master
and a small preview. Install it:

```bash
# macOS
brew install ffmpeg
# Debian/Ubuntu
# sudo apt-get install -y ffmpeg
```

If `ffmpeg` is missing the core tracking still runs, but the final
video-transcode step will fail.

## 3. Directories

Create the directories the code expects:

```bash
mkdir -p models assets _deps
```

## 4. Model weights → `models/`

Download the following into `models/`. The filenames on the left are exactly
what the code expects (the `*` scripts reference these literal paths).

| File (exact name) | Source / how to obtain |
|---|---|
| `face_landmarker.task` | MediaPipe Face Landmarker bundle — https://developers.google.com/mediapipe/solutions/vision/face_landmarker (download the `.task`; it includes the 478-pt mesh + 52 blendshapes + facial-transform head) |
| `pose_landmarker_full.task` | MediaPipe Pose Landmarker (full) — https://developers.google.com/mediapipe/solutions/vision/pose_landmarker |
| `yolov10n-face.pt` | YOLOv10-n **face** detector in Ultralytics format. Obtain an Ultralytics-format YOLOv10-n face checkpoint, or fine-tune/export your own on a face dataset (e.g. WIDER FACE) and save it under this name. The code loads it via `ultralytics.YOLO("models/yolov10n-face.pt")`. |
| `yolov8n-head-scut.pt` | YOLOv8-n **head** detector trained on SCUT-HEAD — https://github.com/Abcfsa/YOLOv8_head_detector (rename the released head-detector weights to this filename). Used as the back-of-head **position** oracle. |
| `6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` | 6DRepNet360 full-range head-pose weights — https://github.com/thohemp/6DRepNet360 (download the `Full-Rotation_300W_LP+Panoptic` checkpoint; keep the name as-is). |
| `buffalo_l/` | InsightFace `buffalo_l` model pack — https://github.com/deepinsight/insightface (only the **optional** BFM-identity experiment uses it; auto-downloaded by InsightFace on first use, or place the pack here). |

> There are no official redistributable checksums for several of these
> third-party weights, so this repo does not pin hashes. Record the SHA-256 of
> whatever you download (`shasum -a 256 models/<file>`) if you need
> reproducibility for your own run.

## 5. Assets → `assets/`

| File (exact name) | What it is |
|---|---|
| `canonical_face_model.obj` | The MediaPipe canonical face mesh (468/478-vertex topology) used as the fallback pose substrate. Export from the MediaPipe canonical face model. |
| `emoji_head.glb` | A glTF/GLB head mesh used as the avatar overlay primitive. Any rigged head GLB with an equivalent topology can be substituted. |

## 6. Optional vendored dependency: 3DDFA_V2 → `_deps/`

Only the optional dense-identity / BFM experiment
(`identity_bfm_solve_v1.py`, `bfm_avatar_overlay_v1.py`) needs this; the core
tracking heads do **not**.

```bash
git clone https://github.com/cleardusk/3DDFA_V2 _deps/3DDFA_V2
# then follow that repo's own build steps (Cython build + its own model/BFM
# downloads). The code expects it at ./_deps/3DDFA_V2.
```

> The Basel Face Model used by 3DDFA_V2 is **academic-use-only**. The BFM path
> in this repo is a research/proof experiment, not a commercially clean asset
> path.

## 7. Input clip

Every pipeline reads a single monocular clip named **`input_clip.mov`** from the
repo root (configurable via the `VIDEO_PATH` constant near the top of each
script). Supply your own front-facing talking-head clip:

```bash
cp /path/to/your/clip.mov ./input_clip.mov
```

No sample clip is included.

---

## 8. What actually runs from a fresh clone — the dependency DAG

**Important:** the three "current heads" are **not** independently runnable from a
fresh clone. Two of them consume intermediate `.npz` streams that must be
produced first by upstream stages. Those `.npz` files are git-ignored and are
**not** in the repo, and there is no single "make" target — you run the stages in
order. The real dependency chain is:

```
input_clip.mov + models/ + assets/
        │
        ▼
pipeline_memoji_rig_v15.py        ─► memoji_rig_stream_v13.npz      (root rig stream)
        │                               └─► consumed by wireframe_overlay_v1.py
        ▼
pipeline_memoji_rig_v17_factored.py ─► memoji_rig_stream_v17.npz
        ▼
pipeline_v17_fusion.py            ─► live_causal_v17_fusion_stream.npz
        ▼
pipeline_mesh_cascade_v2.py       ─► mesh_cascade_v2_stream.npz
        ▼
pipeline_mesh_cascade_v4.py       ─► mesh_cascade_v4_stream.npz   (loads v17_fusion + v2)
        ▼
pipeline_mesh_cascade_v5.py       ─► mesh_cascade_v5_stream.npz   (loads v4)
        ▼
pipeline_mesh_cascade_v6.py       ─► mesh_cascade_v6_stream.npz   (loads v4 + v5)
        ▼
pipeline_mesh_cascade_v7.py       ─► mesh_cascade_v7_stream.npz   (loads v4 + v6)   ◄── HEAD
        ▼
pipeline_predict_correct_v8.py    ─► predict_correct_v8_stream.npz (loads v7 stream) ◄── HEAD
```

Note the deliberately offset filename: `pipeline_memoji_rig_v15.py` writes
`memoji_rig_stream_v13.npz` (the *stream* schema version lags the *script*
version). `wireframe_overlay_v1.py` asserts `pipeline_version=='v15'` on that
file and will refuse a mismatched stream.

### Runnable standalone (only need `input_clip.mov` + `models/` + `assets/`)

These are DAG roots — run them directly after sections 1–7:

- **`pipeline_memoji_rig_v15.py`** — base rig pass (face mesh + 52 blendshapes +
  YOLO face + 6DRepNet360 pose). Produces `memoji_rig_stream_v13.npz`.
- **`pipeline_memoji_rig_v16_headdet.py`** — adds the SCUT-HEAD detector.
  Produces `memoji_rig_stream_v16.npz`.
- **`pipeline_memoji_rig_v17_factored.py`** — factorized-output rig pass.
  Produces `memoji_rig_stream_v17.npz`.

### Need a prior-stage `.npz` (will raise `FileNotFoundError` on a fresh clone)

- **`wireframe_overlay_v1.py`** — needs `memoji_rig_stream_v13.npz`
  (run `pipeline_memoji_rig_v15.py` first) **and** `input_clip.mov`.
- **`pipeline_v17_fusion.py`** — needs `memoji_rig_stream_v17.npz`.
- **`pipeline_mesh_cascade_v2.py` … `v7.py`** — need the chain above, in order.
- **`pipeline_predict_correct_v8.py`** (the headline estimator) — imports
  `pipeline_mesh_cascade_v7` as a module but only *loads* its
  `mesh_cascade_v7_stream.npz`; it does **not** auto-run the upstream stages, so
  the whole chain above must have been run first.

## 9. First command sequence that works from a fresh clone

```bash
# (sections 1–7 done: env, ffmpeg, models/, assets/, input_clip.mov)

# A) Quickest "see it work" path — base rig + a wireframe overlay video:
python pipeline_memoji_rig_v15.py        # -> memoji_rig_stream_v13.npz (+ overlay)
python wireframe_overlay_v1.py           # -> wireframe_tracked_face_master.mp4

# B) Full estimator head (predict -> observe -> correct). Run the whole DAG:
python pipeline_memoji_rig_v17_factored.py   # -> memoji_rig_stream_v17.npz
python pipeline_v17_fusion.py                # -> live_causal_v17_fusion_stream.npz
python pipeline_mesh_cascade_v2.py           # -> mesh_cascade_v2_stream.npz
python pipeline_mesh_cascade_v4.py           # -> mesh_cascade_v4_stream.npz
python pipeline_mesh_cascade_v5.py           # -> mesh_cascade_v5_stream.npz
python pipeline_mesh_cascade_v6.py           # -> mesh_cascade_v6_stream.npz
python pipeline_mesh_cascade_v7.py           # -> mesh_cascade_v7_stream.npz   (HEAD)
python pipeline_predict_correct_v8.py        # -> predict_correct_v8_stream.npz (HEAD)
```

Each stage writes its overlay video, an `.npz` per-frame stream, and a JSON
metrics report to the repo root. The lower-numbered `pipeline_*` /
`mesh_cascade_*` files are retained as the iteration history behind the current
heads; you do not run them in isolation.

See `README.md` for the design-doc index and `MODEL_FUSION_PIPELINE_DESIGN.md`
for the full architecture.
