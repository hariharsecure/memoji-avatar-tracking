# 3DGS Fit Assessment for On-Device Head-Avatar Reconstruction

Scope: a literature review of trusted top-venue 3D Gaussian Splatting (3DGS)
papers for head/face avatars, assessed against this project's constraints:

- **Tracking thread:** a monocular subject clip, Apple Silicon (MPS), no CUDA,
  per-frame head pose + 52 ARKit blendshapes, never-drop overlay.
- **Avatar-rendering thread (downstream):** whether a 3DGS head avatar could be
  driven by the tracker's pose/blendshape stream to produce an identity-stable,
  higher-fidelity rendered avatar than a parametric mesh overlay — and whether
  any of these methods are runnable on Apple Silicon without CUDA.

## Executive Verdict

3DGS is useful as a possible identity-stable avatar representation and renderer.
It is **not** a direct fix for the current tracker, and it is **not** currently
a Mac/MPS-native production path for the trusted head-avatar methods.

For the tracking thread: do not swap the tracker to 3DGS. The existing v15/v16
pipeline already solves the main anchor/smoother issue on Mac/MPS; the remaining
back-of-head HOLD frames are better attacked with a head detector or
segmentation, not a Gaussian avatar. A Gaussian head could be *driven* by the
pose/blendshape stream after reconstruction, but that is a downstream renderer
test.

The hard constraint: the canonical and released research implementations
overwhelmingly depend on NVIDIA CUDA rasterizers, PyTorch3D CUDA,
`diff-gaussian-rasterization`, `simple-knn`, or CUDA-tested setups. Generic
Apple/Metal 3DGS tools exist, but they are not drop-in replacements for
FLAME-rigged head-avatar research pipelines. A higher-fidelity Gaussian renderer
is therefore a bounded cloud/CUDA experiment, not an on-device production path.

## Evidence From Project Notes

- `RESEARCH_INNOVATE_TRACKING.md` identified the core tracking failures as mixed
  anchor references and a Kalman smoother that treated heterogeneous
  measurements as equal.
- `notes_grid_confirm.md` records that v15 fixed this with ear-midpoint anchor
  unification and heteroscedastic Kalman R. The reported state is 847/847
  anchors on head at ear-midpoint level, with 15 unchanged HOLD frames at true
  back-of-head.
- Therefore 3DGS is not needed to fix the current tracking architecture. The
  residual unsolved tracking issue is visibility/position at back-of-head, which
  a head detector or segmentation addresses directly.

## Trusted Paper Table

| Paper | Output | Conditions | Mac/MPS feasibility |
|---|---|---|---|
| Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH/ACM TOG 2023. https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ | Static scene/object radiance field; real-time novel views. Not an animatable face method. | Needs calibrated multi-image/video capture and SfM/COLMAP points. Official optimizer uses PyTorch + CUDA extensions; recommends CUDA GPU, 24GB VRAM for paper-quality training. | Official training is CUDA-gated. Generic Apple/Metal alternatives exist, but they are not the official baseline and not the same as head-avatar pipelines. |
| Qian et al., "GaussianAvatars", CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Qian_GaussianAvatars_Photorealistic_Head_Avatars_with_Rigged_3D_Gaussians_CVPR_2024_paper.pdf | Photoreal drivable head avatar; pose/expression/viewpoint control through FLAME; cross-identity reenactment. | Multi-view videos, known cameras, FLAME tracking, 16 front/side views in NeRSemble; paper reports 600k training iterations. | Not Mac/MPS practical. Released code is research Python built around Gaussian splatting; expected CUDA rasterizer path. Multi-view input also does not match a single monocular capture. |
| Xu et al., "Gaussian Head Avatar", CVPR 2024. https://yuelangx.github.io/gaussianheadavatar/ | Ultra-high-fidelity 2K head avatar with controllable expressions and novel views. | Multi-view RGB videos; 8/16-view sparse-view settings; geometry-guided SDF/DMTet initialization; expression coefficients and head pose from multi-view fitting. | Not fit for Mac/MPS or single-clip data. It explicitly trains under multi-view RGB supervision; not a single-clip or single-reference path. |
| Xiang et al., "FlashAvatar", CVPR 2024. https://ustc3dv.github.io/FlashAvatar/ | Monocular-video digital avatar in minutes; FLAME-embedded Gaussians; claimed >300 FPS at 512 on RTX 3090. | Short monocular video, FLAME tracking, 1-3 min cropped video at 25 fps; paper/repo use NVIDIA RTX 3090 and PyTorch3D. | Best 2024 fit for a cloud-CUDA spike. Not MPS: official repo tested on RTX 3090, installs PyTorch3D and NVIDIA components. |
| Shao et al., "SplattingAvatar", CVPR 2024. https://github.com/initialneil/SplattingAvatar | Full-body/head avatars with mesh-embedded Gaussians; real-time; animated by mesh/blendshapes/skeletal mesh. | Monocular video with a registered mesh template, e.g. FLAME for heads; uses PyTorch CUDA 11.7, PyTorch3D, `diff-gaussian-rasterization`, `simple-knn`. | Strong conceptual fit for "the tracking stream drives a mesh/GS avatar"; implementation is CUDA-gated. |
| Chen et al., "MonoGaussianAvatar", SIGGRAPH 2024. https://github.com/yufan1012/MonoGaussianAvatar | Monocular point/Gaussian head avatar; novel poses/expressions. | Monocular portrait videos, FLAME, IMavatar-style preprocessing; official repo says single NVIDIA 24GB RTX 3090. | CUDA-gated; not Mac/MPS. |
| Ma et al., "3D Gaussian Blendshapes", SIGGRAPH 2024. https://gapszju.github.io/GaussianBlendshape/ | Photoreal head avatar with Gaussian blendshape bases; real-time expression animation. | Monocular video; FLAME 2020; Metrical Photometric Tracker/INSTA preprocessing; official repo tested on RTX 3090/A800 and includes CUDA/C++ viewer. | CUDA-gated; conceptually close to ARKit/FLAME blendshape driving. |
| Dhamo et al., "HeadGaS", ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00280.pdf | Real-time animatable 3D head; expression-aware Gaussian cloud; >10x rendering speedup over baselines. | Monocular moving-head video; needs pose/expression parameters from a parametric head model. | Likely CUDA research path; useful conceptually, not immediate MPS production. |
| Zhou et al., "HeadStudio", ECCV 2024. https://zhenglinzhou.github.io/HeadStudio-ProjectPage/ | Text-to-animatable Gaussian head avatar; novel views and driving by speech/video. | Text prompt + diffusion/distillation + FLAME head prior. Official repo tested with CUDA 11.8 and `diff-gaussian-rasterization`. | Not MPS. Interesting for text-to-avatar experiments, but identity/style control is risky. |
| Giebenhain et al., "NPGA", SIGGRAPH Asia 2024. https://simongiebenhain.github.io/NPGA/ | High-fidelity controllable head avatars using NPHM-conditioned Gaussian dynamics. | Multi-view video recordings, NeRSemble/COLMAP/NPHM tracking; code was listed as coming soon on project page. | Not a single-clip data path; not MPS. |
| Kirschstein et al., "GGHead", SIGGRAPH Asia 2024. https://tobias-kirschstein.github.io/gghead/ | Fast/generalizable 3D Gaussian head generator trained from large 2D image collections; real-time generated heads. | 3D GAN-style model; official repo requires CUDA 11.8 and nvcc to compile Gaussian Splatting kernels. | Not MPS. Interesting as a 3D prior, not a controllable single-identity solution by itself. |
| Teotia et al., "GaussianHeads", SIGGRAPH Asia 2024. https://arxiv.org/abs/2409.11951 | Highly dynamic drivable Gaussian head avatars; mouth interior/teeth/tongue details; real-time. | Carefully calibrated multi-view imagery; coarse-to-fine template mesh + Gaussians. | Not a monocular capture setup; not a Mac path. |
| Tang et al., "GAF", CVPR 2025. https://tangjiapeng.github.io/projects/GAF/ | Animatable Gaussian avatar from short monocular smartphone video; diffusion fills unseen views. | Monocular video, FLAME-based tracking, normal maps, multi-view diffusion pseudo-ground truth; code repo says source code coming soon. | Promising for the monocular case conceptually, but no runnable code path at check time; likely CUDA/diffusion-heavy. |
| Zhang et al., "FATE", CVPR 2025. https://zjwfufu.github.io/FATE-page/ | Monocular full-head 360-degree animatable Gaussian avatar with texture editing and completion. | Single monocular portrait video; completion framework for side/rear head; paper reports single A6000; benchmark lists RTX 3090 x1. | Best conceptual fit for monocular full-head capture if cloud CUDA is available. Not MPS. |
| Feng et al., "GPAvatar", CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Feng_GPAvatar_High-fidelity_Head_Avatars_by_Learning_Efficient_Gaussian_Projections_CVPR_2025_paper.html | Monocular dynamic head avatar; high-dimensional Gaussian projection; high rendering speed and reduced memory. | Monocular video, FLAME tracker; experiments include 2-3 minute and 7 minute monocular videos; RTX 4090 evaluation. | Not MPS; valuable as evidence that monocular Gaussian heads are improving, but still NVIDIA research stack. |
| Wang et al., "MeGA", CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Wang_MeGA_Hybrid_Mesh-Gaussian_Head_Avatar_for_High-Fidelity_Rendering_and_Head_CVPR_2025_paper.html | Hybrid mesh/GS full head; better hair/skin modeling and head editing. | Multi-view videos, NeRSemble, enhanced FLAME face, Gaussian hair; official repo includes CUDA/C++ code. | Not a monocular capture and not MPS; useful design idea for separating face mesh from hair Gaussians. |
| Kirschstein et al., "FlexAvatar", CVPR 2026. https://tobias-kirschstein.github.io/flexavatar/ | Complete animatable 3D head avatar from a single image/few-shot/monocular inputs. | Transformer avatar model trained with partial supervision across monocular and multi-view data; inference can use one portrait. | Most relevant for the single-reference problem, but code/model maturity must be checked before betting. Likely CUDA-class model. |
| Wu et al., "UIKA", CVPR 2026. https://arxiv.org/abs/2601.07603 | Feed-forward animatable Gaussian head from pose-free images, including single image, multi-view captures, smartphone videos. | Uses UV correspondence, synthetic large-scale training data, arbitrary number of unposed inputs. | Relevant watch item if released weights work; not assumed MPS. |
| Ren et al., "OMG-Avatar", CVPR 2026. https://arxiv.org/abs/2603.01506 | One-shot multi-LOD Gaussian head in 0.2s from single image. | Single image; multi-LOD feed-forward model; supplementary mentions restrictive licensing/watermarking. | Potentially relevant for single-image previews, but not a production bet until weights/license/quality are confirmed. |
| Chharia and De la Torre, "MVCHead", CVPR 2026. https://humansensinglab.github.io/MVCHead/ | Multi-view-consistent Gaussian heads learned from 2D images only; single-shot generation. | Trained on random 2D internet images; no multi-view/3D data at training; releases FaceGS-style assets/dataset. | Interesting for 3D priors; too new to treat as an immediate pipeline foundation. |
| Serifi and Bühler, "HyperGaussians", CVPR 2026. https://gserifi.github.io/HyperGaussians/ | More expressive Gaussian representation for animatable face avatars; plugs into FlashAvatar-style pipelines. | Dynamic face avatar videos; improves details such as teeth, glasses, speculars. | Could improve a CUDA FlashAvatar spike; not an MPS unlock. |
| Kabadayi et al., "PhysHead", CVPR 2026. https://phys-head.github.io/ | Simulation-ready Gaussian head avatars with dynamic hair. | Multi-view video, FLAME/hair mesh, Maya hair simulation; repo currently exposes sample/simulation pieces with avatar reconstruction TODO. | Not useful for the current Mac pipeline; future hair-design insight only. |

## Fit Assessment

### Tracking Thread

3DGS does not solve the current tracker. The tracker outputs pose/blendshapes;
3DGS head avatars *consume* pose/expression signals after reconstruction. They do
not replace YOLO/MediaPipe/6DRepNet/ear-anchor/IMM-Kalman for never-drop
tracking on a monocular clip.

The remaining tracking issue is back-of-head HOLD frames. 3DGS can render a
back-of-head *if* a full-head avatar exists, but it cannot infer the live head
location from a frame where the face is invisible unless a separate
tracker/segmenter/detector provides that location. A head-class detector,
silhouette segmenter, optical flow, or body-pose head proxy is the direct fix.

### Avatar Renderer (downstream)

Best honest path: a CUDA/cloud spike, not Mac production.

Use a clean 1-3 minute subject capture with varied yaw/pitch/expression and
stable lighting. Reconstruct with FlashAvatar/FATE/GPAvatar/SplattingAvatar-class
code on NVIDIA. Then test driving it from the v15/v16 per-frame pose + 52 ARKit
stream after mapping ARKit -> FLAME/expression parameters.

Expected benefit: more identity-consistent and realistic than a parametric mesh
overlay.

Expected risk: training/setup friction, FLAME tracker mismatch, hair/back-of-head
artifacts, ARKit-to-FLAME mapping loss, and no MPS-native reproducibility.

### Stylized-Character Animation Thread

For animating a *stylized* (non-photoreal) virtual character rather than a
photoreal human, 3DGS helps only if it becomes a stable 3D identity/control
layer. It does not directly produce a stylized look. All photoreal head-avatar
papers optimize toward photoreal human capture; for a stylized target the value
of a 3D Gaussian/mesh prior is geometric consistency, not final appearance.

A plausible direction:

1. Use a one-shot/few-shot Gaussian head model (or a generated multi-view
   identity grid) to build a stable 3D head prior for the target character.
2. Render consistent pose/depth/normal/landmark/control frames from that prior.
3. Use those frames as ControlNet/IP-Adapter/identity constraints for a
   downstream stylizer.
4. Gate results by cross-shot identity consistency before committing to
   animation.

Risk: one-shot Gaussian head models may not preserve a stylized identity, may
require unavailable weights, and may render in a photoreal/3D-portrait regime
that does not match the target style.

## Where 3DGS Fits in This Project

1. **Never-drop tracking as the driver.** Most 3DGS avatar papers assume their
   own FLAME tracker. This project already has a Mac-runnable stream of per-frame
   pose and 52 ARKit blendshapes. If ARKit can be mapped to FLAME, the tracker
   becomes a reusable driver for any reconstructed avatar.

2. **3DGS as identity-stable control, not final appearance.** A 3D head prior
   provides stable identity and multi-view geometry; it does not replace the
   appearance/stylization stage.

3. **Deterministic control stack.** Mesh/landmarks/depth/normals/pose from the
   rig can become ControlNet-style conditioning, which is more controllable than
   unconstrained generation.

4. **Mac-first, CUDA-only spike.** Keep production tracking, render-control, and
   evaluation on Apple Silicon. Use cloud CUDA only for bounded reconstruction
   experiments. Do not contaminate the MPS production pipeline with CUDA-only
   dependencies.

5. **Objective gates.** Identity consistency, temporal flicker, mouth-shape
   legibility, tracker jitter, and held-out pose reenactment can all be measured
   before committing.

## Candidate Testable Hypotheses

**H1: 3DGS is not needed for the tracker.**
Test: add a YOLO/segmentation head-anchor lane for the 15 HOLD frames before
trying any 3DGS. Success: HOLD frames drop from 15 to <=5 and avatar placement
remains head-level. Failure: no HOLD improvement.

**H2: A CUDA FlashAvatar/FATE-style avatar can be driven by the tracker stream.**
Test: capture 1-3 minutes of a subject with varied expressions/yaw; reconstruct
on NVIDIA; map the ARKit stream to FLAME/expression controls; render the existing
847-frame clip. Success: same identity across frames, self-reenactment visually
tracks mouth/head, >30 fps render after training. Failure: expression mismatch,
hair/back artifacts, setup cannot train in one day.

**H3: Mac/MPS cannot run trusted head-avatar 3DGS training without a port.**
Test: attempt install of one representative repo, e.g. FlashAvatar or
SplattingAvatar, on MPS. Success condition for "Mac viable": no CUDA/NVIDIA
packages and training one subject completes. Expected result: fail at
PyTorch3D/CUDA/diff-gaussian-rasterization. A generic OpenSplat/Metal success
does not count as head-avatar success.

**H4: A stylized target needs a 3D identity prior plus a controlled stylizer,
not unconstrained generation over a bare mesh.**
Test: build a 6-pose control grid from a one-shot/few-shot Gaussian head or
generated multi-view prior; run a stylizer with identity/structure conditioning.
Success: review says same character in all 6 poses with no structural
hallucination. Failure: identity drifts or structure is not preserved.

## Decision

Do not make 3DGS a core dependency of the Mac/MPS tracking pipeline now.

Run one bounded CUDA/cloud spike for avatar reconstruction only if the goal is a
higher-fidelity, identity-stable renderer than a parametric mesh overlay.

Treat 3DGS as a possible 3D identity/control prior for downstream stylized
animation. It should enter the pipeline only if it beats the current
single-mesh control render on identity consistency without reintroducing
structural hallucination.
