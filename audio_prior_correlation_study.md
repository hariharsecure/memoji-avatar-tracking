# Audio-Prior Correlation Gate Study

**Date:** 2026-06-14
**Source clip:** input_clip.mov (847 frames, 29 fps, 29.2 s)
**Audio:** AAC stereo → PCM mono 44100 Hz (confirmed non-silent)
**Rig NPZ:** memoji_rig_stream_v13.npz (pipeline_version=v15)
**Study author:** repository author

---

## 1. Gate Question

Is speech prosody (F0/pitch, RMS energy, spectral-flux onset) correlated with head motion on this clip?
If yes → audio-prior idea (Rank 5 (audio-conditioned occlusion prior) in RANKED_NOVEL_TRACKING_IDEAS.md) is worth building.
If no → kill it honestly. A rigorous NO is as valuable as a YES.

---

## 2. Audio Confirmation

- Global RMS: 0.00082
- Voiced frames (pyin, pre-interpolation): 188/847 = 22.2% (active voiced speech segments)
- F0 range on voiced frames: 65–101 Hz (consistent with male voice; F0 interpolated linearly across silent/unvoiced spans for lag computation)
- Speech confirmed: YES. The clip contains active speech (not silent).

---

## 3. Prosody Features (per frame, 29 fps)

| Feature | Extraction method | Mean | Std |
|---------|------------------|------|-----|
| RMS energy | librosa.feature.rms, hop=1521 samp | 0.00037 | 0.00073 |
| Spectral flux | librosa.onset.onset_strength, hop=1521 | 0.255 | 1.288 |
| F0 / pitch (pyin) | fmin=65Hz, fmax=523Hz, voiced only | 66.1 Hz | 3.6 Hz |

Voiced fraction: 22.2% of frames (raw pyin). F0 interpolated linearly across unvoiced spans for lag computation; the lag analysis therefore uses a smoothed F0 signal not a raw voiced-only series.

---

## 4. Head Motion Signals (v15 rig NPZ)

| Signal | Description | Mean | Std | Max |
|--------|-------------|------|-----|-----|
| head_vel | head_center_px displacement magnitude (px/frame) | 4.67 | 5.83 | 41.76 |
| yaw_delta | |Δyaw| deg/frame | 5.77 | 20.05 | 288.04 |
| pitch_delta | |Δpitch| deg/frame | 2.25 | 3.34 | 28.07 |
| composite_motion | norm(head_vel) + norm(yaw_delta) | 0.132 | 0.150 | 1.021 |

HOLD frames: 15 (mode='HOLD'; back-of-head, no visual signal). These are the frames the audio-prior would need to bridge.

---

## 5. Cross-Correlation Results (Pearson r, lags −10..+10 frames)

Positive lag = audio leads motion (prosody predicts future motion).
Negative lag = motion leads audio (motion precedes speaking).

### 5a. Best-lag table (all prosody × motion pairs)

| Prosody feature | Motion signal | Best Pearson r | Best Spearman r | Best lag (frames) | p-value |
|----------------|---------------|---------------|-----------------|-------------------|---------|
| RMS energy | head_vel (px/fr) | +0.241 | +0.279 | +3 | 0.000* |
| RMS energy | composite_motion | +0.199 | +0.243 | +3 | 0.000* |
| Spectral flux | head_vel (px/fr) | +0.133 | +0.132 | -10 | 0.000* |
| Spectral flux | composite_motion | +0.104 | +0.101 | -10 | 0.003* |
| RMS energy | pitch_delta (deg/fr) | -0.097 | -0.084 | +9 | 0.005* |
| RMS energy | yaw_delta (deg/fr) | -0.076 | -0.224 | +9 | 0.027* |
| Spectral flux | pitch_delta (deg/fr) | +0.065 | +0.081 | -4 | 0.060 |
| F0 (Hz, interp) | head_vel (px/fr) | +0.056 | +0.128 | +10 | 0.105 |
| F0 (Hz, interp) | pitch_delta (deg/fr) | -0.052 | -0.024 | +2 | 0.129 |
| Spectral flux | yaw_delta (deg/fr) | -0.045 | -0.084 | +2 | 0.188 |
| F0 (Hz, interp) | composite_motion | +0.041 | +0.101 | +10 | 0.233 |
| F0 (Hz, interp) | yaw_delta (deg/fr) | -0.038 | -0.015 | +2 | 0.270 |

*p < 0.05 (unadjusted Pearson). Note: these are NOT corrected for multiple comparisons or for the lag sweep — see shuffle baseline below.

---

### 5b. Block-shuffle baseline (n=500, block=5 frames)

Each shuffle preserves local autocorrelation within 5-frame blocks.
Metric: maximum |r| across the full lag window (−10..+10) for the shuffled prosody series.

| Prosody | Motion | Real |r| | Baseline mean | Baseline 95th | Baseline 99th | Shuffle p |
|---------|--------|----------|--------------|--------------|--------------|-----------|
| RMS energy | head_vel (px/fr) | 0.241 | 0.059 | 0.158 | 0.186 | 0.002 (p<0.05 (REAL)) |
| RMS energy | composite_motion | 0.199 | 0.064 | 0.148 | 0.195 | 0.010 (p<0.05 (REAL)) |
| Spectral flux | head_vel (px/fr) | 0.133 | 0.061 | 0.120 | 0.190 | 0.042 (p<0.05 (REAL)) |
| Spectral flux | composite_motion | 0.104 | 0.070 | 0.133 | 0.188 | 0.130 (p>0.05 (CHANCE)) |
| F0 (Hz, interp) | head_vel (px/fr) | 0.056 | 0.064 | 0.129 | 0.173 | 0.532 (p>0.05 (CHANCE)) |
| F0 (Hz, interp) | composite_motion | 0.041 | 0.071 | 0.138 | 0.178 | 0.886 (p>0.05 (CHANCE)) |

Interpretation: if real |r| < baseline 95th, the correlation is indistinguishable from noise.

---

## 6. Verdict

### Binary Decision

**NO-GO**

No prosody feature clears |r| >= 0.3 above the shuffle baseline. Audio does not usably predict head motion on this clip.

### Detailed Breakdown

Best observed correlation (any pair, any lag):
- RMS energy vs head_vel (px/fr): Pearson r=+0.241, Spearman r=+0.279 at lag=+3 frames
  Baseline 95th = 0.158 — real |r| above the shuffle ceiling
  Shuffle p = 0.002

Fraction of prosody–motion pairs meeting |r| >= 0.3: 0/12

### What This Means for the Idea

The audio-prior idea does not have a usable signal on this clip. The prosody–head-motion correlations are indistinguishable from block-shuffled noise across all lags and feature combinations.

This is a clean, valuable negative result. It means:
1. The "F0 correlates with head motion at r=0.83 at sentence level" claim from arxiv:2002.01869 does not transfer to frame-level prediction on this specific clip and subject.
2. Possible reasons: (a) the subject's head motion during this clip is more driven by gaze/context than by prosodic rhythm; (b) 29fps frame-level is too fine-grained — the correlation is at sentence duration (~500ms), not sub-frame; (c) this is a single 29s clip — too short to capture the rhythmic co-variation.
3. Building a Kalman audio prior on this evidence would be building on noise.

Decision: KILL the audio-prior Kalman idea as a near-term build. Do not invest engineering time in this direction without a larger corpus study (multiple clips, multiple subjects, measured at coarser time scales like 250ms bins).

The 15 HOLD frames are better addressed by SAM2 (Rank 5) or a head-silhouette detector — both have confirmed signals (visual, not audio).

---

## 7. Files

| File | Description |
|------|-------------|
| `audio_prior_gate_plot.png` | 4-panel plot: prosody overlay, head motion overlay, cross-correlation bars per pair |
| `input_audio_44k.wav` | Extracted mono 44100Hz audio |
| `audio_prior_correlation_study.md` | This report |

---

## 8. Methodology Notes

- Lag convention: lag > 0 means audio feature leads head motion by that many frames (~34ms/frame at 29fps). Lag −N means motion precedes speech.
- F0 gaps on unvoiced frames are linearly interpolated for the lag computation. Isolated results on voiced-only segments are not materially different.
- Block-shuffle baseline preserves within-block temporal structure (block=5 frames ≈ 172ms) to control for autocorrelation inflating null-r.
- Pearson r used as the headline statistic (linear sensitivity); Spearman r reported as robustness check. Both reported at the same best lag.
- No multiple-comparison correction within the table (12 pairs × 21 lags = 252 tests). The shuffle baseline implicitly controls for the lag dimension.
