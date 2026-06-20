# Avatar v8 — Predict-Correct State Estimator (design)

**Design note — specifies the v8 estimator before implementation.**

## The reframe
The core premise: **the "floor" is not unknowable — it is unverified.** Everything about a talking
head is correlated, not magic — head geometry, the next mouth shape, the next head position from
body motion are each *predictable from signals we can observe*. So we stop treating hidden
quantities as a hard wall and instead **predict every hidden quantity from correlated signals as a
confidence-weighted prior, then correct it when a real observation verifies it.** The prediction
*is* the generalization; the observation *is* the verification.

This makes the avatar a **confidence-weighted recursive state estimator** (predict → observe →
correct) — not another one-off smoother like v6/v7.

## State (each carries a confidence / covariance)
- **HeadPose** — position, velocity, quaternion rotation, angular velocity
- **Geometry** — fixed identity / head shape + slow canonical-mesh residuals (the pure-math head model)
- **Expression** — ARKit-52 blendshapes + velocities + a mouth/viseme latent
- **Appearance** — UV-atlas texture, per-texel confidence + age + source view-angle
- **Body** — shoulders / torso / neck pose + velocity *(new)*

## Predict (generalize from coupled signals)
- **Pose** = v7 quaternion constant-angular-velocity + optical-flow head-anchor **+ body→head
  kinematic prior** — torso yaw / shoulder line / neck base predict the probable next head
  center+yaw when the face is weak. *Soft prior, not truth.*
- **Expression / mouth** = temporal blendshape dynamics + **audio→viseme** + co-articulation delay
- **Geometry** = rigid / slow; accumulate only from high-confidence verified frames
- **Texture** = persistent UV-atlas + temporal stability + low-confidence bilateral-symmetry fill for the far side

## Observe
Mesh cascade (verified face/mesh) · optical flow (visible-head motion) · body-pose detector
(MediaPipe Pose — torso/shoulder/neck) · audio (viseme *likelihood*, not truth) · UV updates only
from visible, well-projected texels.

## Correct (verify)
- Pose: **error-state Kalman on SO(3)** (rotation) + linear KF (position / velocity)
- Expression: robust Bayesian filter over blendshapes
- Geometry: slow robust update, verified frames only
- Texture: per-texel Bayesian / EMA with confidence + view-angle + age
- **Innovation gating throughout:** if an observation contradicts the prediction too strongly,
  down-weight or reject it — *never blindly snap* (the v6/v7 failure mode)
- **Every output quantity is tagged `verified` / `predicted` / `mixed`; a prediction is NEVER
  promoted to verified without an observation.** ← "generalizable until verified" made literal.

## Buildable now vs stretch (honest)
**Now (Mac / MPS, CUDA-free):** pose estimator (v7 CAV + flow + body→head prior) · MediaPipe-Pose
body prior · expression temporal filter + simple audio-viseme · per-quantity confidence fields ·
innovation + calibration report · basic UV atlas with per-texel confidence.
**Stretch (predict but low-confidence — may be *confidently wrong*):** reliable hidden-mouth from
audio alone · far-side texture under asymmetric light/expression · body→head across *independent*
head turns (head turns while body doesn't) · high-quality persistent full-head identity without a
stronger 3D head model.

## Expectation
**What should improve:** fewer pose snaps in face-weak / profile / back spans · better hidden-span
head continuity from body motion · less frozen mouth during occlusion (speech-like motion) ·
steadier identity across detector modes · far-side becomes *plausible + confidence-marked*, not "truth."

**Pre-registered metrics:** pose angular-jump p99 + f425–440 max step + post-hidden reacquisition
innovation · mesh vertex-step p95 over head-scale in hidden spans · body-prior head-center/yaw
error at next verified frame vs v7 baseline · hidden-span jaw/lip error vs next verified mouth +
viseme timing error · texture coverage + far-side color error when later observed · **calibration:
predicted-confidence vs actual-error (reliability / ECE).**

**Kill / falsifier conditions (pre-registered):**
- body→head prior *increases* reacquisition error vs the v7 CAV/flow baseline
- audio-viseme prior is confidently-wrong more often than a hold / constant-velocity expression baseline
- symmetry prior fails on asymmetric expression / wink / squint / occluded far side *while claiming high confidence*
- geometry slow-update causes identity drift or shimmer
- calibration fails — low-error and high-error predictions receive similar confidence
- **any predicted region is rendered or exported as `verified` without an observation**

**Honest bottom line:** this does **not** make hidden truth magically correct. It makes hidden
spans **smoother, more correlated with real motion / speech / body dynamics, and honestly
confidence-labeled until verified.** Where verification never comes, the output stays a *labeled
prediction* — plausible, never claimed as truth.
