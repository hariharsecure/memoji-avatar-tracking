# Ranked Novel Tracking / Avatar Research Ideas

Scope: the current avatar tracking stack in this repository, especially `RESEARCH_INNOVATE_TRACKING.md`, `notes_grid_confirm.md`, `notes_live_causal.md`, `wireframe_accuracy_report.md`, `NOVELTY_TRANSFER_ASSESSMENT.md`, and `3DGS_HEAD_AVATAR_ASSESSMENT.md`.

## Baseline I Am Not Counting As Novel

The current stack is a competent integration, not an algorithmic novelty claim:

- v15 already uses ear-midpoint anchor unification and heteroscedastic Kalman observation noise.
- live causal v1 already uses a forward IMM Kalman and is roughly 24 fps-class on M3 Ultra, not proven 29/30 fps source-rate live.
- wireframe output is measured, but only 44% of frames are true MediaPipe primary mesh; fallback frames are pose/position-level.
- 3DGS is useful as an avatar renderer/reconstruction direction, not as a direct Mac/MPS tracking fix.

The ranking below treats those as baseline engineering.

## Genuinely Novel / Open Research Hypotheses

These are not "proven first." They are the ideas where I did not find exact prior art for this specific problem formulation. The primitives are known, so the honest claim is: plausible novel problem framing plus testable implementation — not a verified novelty claim.

### 1. Cross-Modal Self-Distilled Fallback State Model

Core insight:
Train a tiny per-subject sequence model to predict the canonical head state from degraded channels: body-pose ears/shoulders, head box or silhouette, optical flow, previous state, audio prosody, detector confidence, and optionally render residual. Training signal comes from visible frames where MediaPipe face/mesh provides a high-quality teacher. During training, randomly hide the face channel so the model learns exactly the failure mode: "what should the canonical face/head state be when the face detector drops?"

Why this may be open:
Self-supervised facial landmark refinement via video registration exists: Supervision-by-Registration uses optical-flow coherence with no extra labels. Person-specific online face alignment exists. Self-supervised adaptation of high-fidelity face models from commodity camera video exists. But those are mostly improving a face model or landmark detector when the face is still observable. I did not find exact prior art for a small causal avatar tracker that learns a cross-detector fallback translator from visible overlap and then uses it during face absence/back-of-head/profile gaps. This is still "couldn't find exact prior art," not proof of novelty.

Mac/MPS path:
Pure PyTorch MPS. Tiny GRU/TCN/Transformer encoder, 10-50k parameters to start. Inputs are already available from the current pipeline plus optional audio RMS/MFCC. No CUDA, no new heavy model.

Testable PoC:
Use visible frames as teacher labels. Create synthetic dropouts matching real profile/back-of-head periods. Train on 80% of visible spans, test on held-out visible spans and the real 15 HOLD/profile frames by manual head-center labels. Success: at least 30% lower anchor error than current pose/HOLD fallback during synthetic dropouts, uncertainty correlated with error, no >50 px new jumps. Kill condition: no improvement over current IMM + source-aware R on held-out synthetic dropouts, or confidence is uncalibrated.

### 2. Risk-Bounded "Never Lie" Avatar Controller

Core insight:
Stop treating "never-drop" as "always output a confident point." The tracker should output a belief set: position ellipse, pose distribution, blendshape validity, and mode trust. The avatar controller then chooses an honest rendering action: normal drive, damped drive, pose-hold, expression-decay, partial occlusion, or explicit low-confidence visual state. The innovation is not uncertainty alone; it is a calibrated contract between tracker and avatar behavior.

Why this may be open:
Uncertainty-aware pose estimation exists, including probabilistic angular regression with von Mises mixtures. Conformal prediction for bounding boxes/keypoints/pose uncertainty also exists. v15 already uses heteroscedastic R. I did not find exact prior art for an avatar tracking controller that exposes calibrated per-frame uncertainty and optimizes the render policy against perceptual avatar failures rather than only tracking RMSE. This is a product/research formulation, not a new UQ algorithm.

Mac/MPS path:
Mostly NumPy/PyTorch CPU/MPS. Add calibration tables from held-out annotations, conformal residual quantiles by mode/source/yaw/pitch, and renderer policy logic.

Testable PoC:
Manually annotate 150-300 frames across 20 clips. Calibrate 90% coverage ellipses by detector mode and pose bucket. Run A/B videos: current always-on avatar vs uncertainty-contract avatar. Success: empirical coverage within +/-5% of target and human review prefers the uncertainty-aware render on failure zones without worse normal frames. Kill condition: coverage misses badly, or users prefer the current deterministic overlay because uncertainty actions are visually distracting.

### 3. Personalized Skull/Neck Latent Prior for Occluded Head State

Core insight:
Fit a tiny subject-specific kinematic model of head center, neck pivot, shoulder frame, and feasible yaw/pitch/roll dynamics from the subject's own visible video. Use it as a physical prior when the face disappears. The state is not "face landmarks"; it is "hidden skull pose relative to torso." This reframes back-of-head/profile tracking as constrained latent-state estimation, not detector fallback.

Why this may be open:
Biomechanical and anatomical pose priors exist. Upper-body pose priors have been learned from anatomy/biomechanics/physics constraints rather than mocap. BioPose-style monocular pose methods now include biomechanical skeleton models. Person-specific face alignment also exists. I did not find a compact, Mac-runnable, avatar-specific skull/neck prior learned from a single user's video and fused with face/body detectors for face-absent frames. Again: plausible open formulation, not proven first.

Mac/MPS path:
No heavy model needed for v0. Fit parameters by nonlinear least squares or small neural state-space model: neck pivot offset, head radius/scale, per-subject motion limits, yaw/pitch coupling, shoulder-to-head constraints. Runs CPU/NumPy; optional tiny PyTorch MPS model.

Testable PoC:
Fit on visible frames and body pose. Evaluate on synthetic occlusions plus real profile/back-of-head spans. Success: lower error than body-pose ears alone and fewer avatar scale/position artifacts at profile/HOLD. Kill condition: shoulder/pose noise dominates so the prior adds lag or false confidence; no measurable improvement on annotated clips.

## Novel Application Of Known Methods

### 4. Render-and-Compare Residual Tracker

Core insight:
Use the avatar or wireframe as an active sensor. Render the predicted mesh/avatar/head silhouette into the frame, compare against image edges, segmentation, optical flow, or face mesh when visible, and feed the residual back into the state estimator.

Novelty assessment:
Analysis-by-synthesis and 3DMM face fitting are old. Differentiable rendering, DECA/EMOCA, IMavatar, Neural Head Avatars, and Gaussian avatars all undercut broad novelty. The possible novelty is only the lightweight causal use: a Mac-compatible residual corrector around the existing avatar driver, not a full neural avatar reconstruction.

Mac/MPS path:
Start non-differentiable: render wireframe/silhouette with OpenCV/Metal, compute residual features, run a small Gauss-Newton or learned MLP correction. Avoid CUDA differentiable rasterizers.

Testable PoC:
On primary frames, deliberately perturb anchor/pose and see if residual descent recovers the original. Then test profile/HOLD frames with a silhouette/head mask. Success: residual correction reduces manual error without oscillation. Kill condition: residual is ambiguous under lighting/hair/background and worsens stable frames.

### 5. Audio-Conditioned Occlusion Prior

Core insight:
Use speech audio to predict likely low-frequency head motion during face dropout, but only as a weak prior inside the filter. It should never override vision when vision is confident.

Novelty assessment:
Audio-driven head motion is already a full research area. Audio2Head explicitly predicts rigid 6D head movement from speech. The novel application would be using audio only as an uncertainty-gated prior for occluded tracking, not generating a talking-head video.

Mac/MPS path:
CPU audio features plus tiny GRU/TCN on MPS. Inputs: MFCC/RMS/pitch, previous pose, maybe speaker-specific style embedding. Output: delta yaw/pitch/roll and confidence.

Testable PoC:
Train per-subject on visible speaking segments. Mask visual pose for 1-2 seconds and compare audio prior vs constant-velocity/IMM. Success: lower orientation drift during speaking occlusions without harming silent spans. Kill condition: audio predicts average nodding style but not actual motion; no gain over biomechanical/IMM prior.

### 6. Learned Mode-Arbitration / Never-Drop Policy

Core insight:
Replace the hand-tuned cascade with a learned policy that chooses among MediaPipe, YOLO+6DRepNet, body-pose, head detector/silhouette, optical-flow propagation, HOLD, or "degrade avatar." Reward should penalize jumps, identity switches, false confidence, and avatar-visible failures.

Novelty assessment:
Learning data association and tracking policies is established. MDP/RL tracking and multi-agent RL MOT both exist. The novel application is a single-avatar detector-mode policy optimized for render quality and uncertainty, not general MOT.

Mac/MPS path:
Start as supervised imitation/logistic ranking from existing pipeline plus manual labels, not RL. Tiny classifier or bandit over mode features. MPS not required.

Testable PoC:
Build a mode-choice dataset from 20 clips with manual "best source" labels on failure frames. Success: fewer visible jumps and false mode choices than the hand cascade. Kill condition: labels too sparse/noisy; policy overfits one clip and fails on multi-person distractors.

### 7. Head-Silhouette / Back-of-Head Tracker

Core insight:
Treat back-of-head as a silhouette/object boundary problem, not a face problem. Use a head segmentation or ellipse/active-contour tracker to maintain the skull center when face landmarks disappear.

Novelty assessment:
Head detection/tracking, segmentation, active contours, 3D model-based tracking, and crowd head datasets are established. The useful application is making the remaining 15 HOLD frames measurement-bearing on Mac, not inventing head tracking.

Mac/MPS path:
Use an open segmentation/head detector if available on MPS, or start with lightweight GrabCut/SAM-style mask if dependency cost is acceptable. Fit ellipse/head contour and fuse as low-confidence position-only measurement.

Testable PoC:
Evaluate only the real HOLD/profile windows first. Success: HOLD frames drop from 15 to <=5 or position error improves by >25 px without false positives. Kill condition: hair/background/shoulders make the contour unstable or detector misses back-of-head.

### 8. Tiny Per-Subject Test-Time Anchor Adapter

Core insight:
Fine-tune a small correction head online/person-specifically so detector outputs map to the subject's stable avatar anchor. It learns that subject's face shape, ear/shoulder geometry, camera scale, and systematic detector biases.

Novelty assessment:
Person-specific online face alignment exists, and self-supervised face-model adaptation exists. The novelty is only the target: a tracker-anchor correction layer for this Mac avatar stack.

Mac/MPS path:
Freeze all heavy detectors. Train a tiny MLP/linear model on features: detector source, landmarks, bbox, yaw/pitch/roll, scale, confidence, temporal derivatives. CPU/MPS.

Testable PoC:
Train on first N visible frames, test on later frames/manual labels. Success: reduces pose_calib RMSE and profile anchor bias without drift. Kill condition: adapter learns early clip bias and worsens later close-up/profile frames.

### 9. Tracker-Derived Dense Control For Avatar/Video Stylization

Core insight:
Use the tracked mesh/wireframe/depth/normal/head-pose stream as deterministic control input for a renderer or diffusion stylizer, so the avatar stays identity/pose-consistent across shots.

Novelty assessment:
Face-landmark ControlNet exists and directly uses facial landmarks as conditioning. ControlNet-style structured conditioning is known. The application to a consistent tracker stream may help an avatar/stylization workflow, but it is not tracking innovation.

Mac/MPS path:
Control image generation can be local only if models fit MPS; otherwise use bounded external/CUDA spikes. The control stream itself is already Mac-runnable.

Testable PoC:
Render 6 pose/expression control frames from the same tracker stream and pass through low-denoise stylization. Success: identity consistency improves over stills-first generation. Kill condition: stylizer ignores controls or introduces mesh/domain artifacts.

## Already Done / Do Not Pitch As Innovation

### 10. Gaussian Splat As The Core Tracking Solution

Assessment:
Already done as head-avatar representation: GaussianAvatars, FlashAvatar, MonoGaussianAvatar, Gaussian Blendshapes, FATE, GPAvatar, MeGA, and related work. It can improve avatar fidelity after reconstruction, but it does not solve detector dropout or live head localization by itself. The trusted implementations are mostly CUDA-gated.

Mac/MPS:
Not a production Mac/MPS tracking path today. Use only as a bounded cloud/CUDA renderer/reconstruction spike.

Test / kill:
Only test if the goal is avatar fidelity. Kill as tracker fix if setup cannot infer head position during face-absent frames without a separate detector.

### 11. Wireframe-As-Tracking-Substrate, If It Means "Use Mesh Landmarks"

Assessment:
Already done in 3DMM, active appearance/shape models, MediaPipe/FaceMesh, FLAME trackers, DECA/EMOCA/IMavatar/NHA-style pipelines. The only worthwhile version is rank #4: render residual as a causal correction signal in this specific stack.

Mac/MPS:
Mesh landmark extraction already exists locally. Full differentiable face fitting is heavier and often CUDA-oriented.

Test / kill:
Do not build a second generic FLAME tracker on Mac unless it beats v15 on annotated frames. Kill if it is slower and not more accurate than the current anchor stream.

### 12. Uncertainty As A First-Class Output, If It Means "Add Confidence/R Values"

Assessment:
Already done broadly, and v15 already implements source-aware observation noise. Probabilistic pose estimation and conformal bounding-box/keypoint uncertainty exist. The only research-worthy version is rank #2: calibrated uncertainty that changes avatar behavior and is evaluated against coverage/perception.

Mac/MPS:
Already cheap.

Test / kill:
Kill any "uncertainty" work that does not produce calibrated coverage or better avatar decisions.

### 13. Learned Tracking Policy, If It Means Generic RL/MOT

Assessment:
Already done. MDP/RL tracking and learned data association have long prior art. Only the avatar-mode policy with render-quality reward is a novel application.

Mac/MPS:
Tiny policy works locally; RL training is unnecessary until supervised labels fail.

Test / kill:
Kill if a simple calibrated heuristic matches the learned policy.

### 14. Mesh/Landmarks-As-ControlNet

Assessment:
Already done. Face-Landmark-ControlNet is a direct prior example. This is a renderer/stylization control path, not a new tracker.

Mac/MPS:
Possible for small diffusion models, but not core tracking.

Test / kill:
Kill if it does not preserve identity better than existing deterministic avatar rendering.

## Ranked Recommendation

1. Build a small offline research PoC for the cross-modal self-distilled fallback state model.
2. In parallel, add calibrated uncertainty artifacts to the current tracker and test the "never lie" avatar policy.
3. Fit a personalized skull/neck prior only after manual annotations exist, because otherwise success cannot be measured.
4. Treat render-and-compare as a contained experiment, not a rewrite.
5. Keep Gaussian splats and ControlNet out of the tracking novelty claim; they belong to avatar fidelity/stylization, not head tracking.

## Source Pointers Used

- 3DMM analysis-by-synthesis: https://pure.mpg.de/rest/items/item_3255852/component/file_3327797/content
- Supervision-by-Registration: https://github.com/facebookresearch/supervision-by-registration
- Self-supervised high-fidelity face model adaptation: https://arxiv.org/abs/1907.10815
- Sequential/person-specific face alignment: https://openaccess.thecvf.com/content_cvpr_2016_workshops/w28/papers/Peng_Sequential_Face_Alignment_CVPR_2016_paper.pdf
- Deep Directional Statistics / pose uncertainty: https://arxiv.org/abs/1805.03430
- Conformal bounding-box uncertainty: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12292.pdf
- Learning-to-track / MDP tracking: https://cvgl.stanford.edu/papers/xiang_iccv15.pdf
- Multi-agent RL MOT: https://ifaamas.org/Proceedings/aamas2018/pdfs/p1397.pdf
- MOT data association survey: https://arxiv.org/abs/1802.06897
- Audio2Head: https://arxiv.org/abs/2107.09293
- Face-Landmark-ControlNet: https://github.com/Georgefwt/Face-Landmark-ControlNet
- GaussianAvatars: https://github.com/ShenhanQian/GaussianAvatars
- BioPose / biomechanical monocular pose: https://arxiv.org/html/2501.07800v1
- Data-free upper-body biomechanics prior: https://pubmed.ncbi.nlm.nih.gov/23893728/
- IMavatar: https://arxiv.org/abs/2112.07471
- Neural Head Avatars: https://openaccess.thecvf.com/content/CVPR2022/papers/Grassal_Neural_Head_Avatars_From_Monocular_RGB_Videos_CVPR_2022_paper.pdf
- DECA: https://github.com/yfeng95/DECA
