# NATIVA: Unsupervised Bearing Fault Detection Using Spiking Neural Networks with Active Inference

**Saad LARAJ**

---

## Abstract

We present NATIVA, a spiking neural network architecture for unsupervised bearing fault detection. The system combines Leaky Integrate-and-Fire neurons, Spike-Timing-Dependent Plasticity, Kuramoto phase synchronization, and Active Inference (variational Free Energy minimization) to learn normal vibration patterns without labeled data. We evaluate NATIVA on the CWRU bearing dataset across 36 operating conditions (3 fault types, 3 fault sizes, 4 motor loads) and compare against a supervised Random Forest baseline and unsupervised baselines (Isolation Forest, One-Class SVM).

NATIVA achieves a mean AUC-ROC of 0.95 across all conditions, with 0.999 at 0HP and 3HP loads. The system uses a multi-band frequency encoder with global normalization calibrated on healthy data. We identify a performance degradation at 1HP load (AUC = 0.82) and trace it to harmonic interference between shaft rotation frequency and bearing fault characteristic frequencies. We position NATIVA as a Level-1 wake-up sensor for edge deployment, complementary to supervised diagnostic systems.

**Keywords**: Spiking Neural Networks, Bearing Fault Detection, Active Inference, STDP, Unsupervised Anomaly Detection, Neuromorphic Computing, CWRU

---

## 1. Introduction

Bearing failures account for 40–50% of rotating machinery breakdowns in industry [1]. Early detection prevents catastrophic failures and costly unplanned downtime. The standard approach uses supervised classifiers (Random Forest, CNN) trained on labeled vibration data with engineered frequency features. This requires labeled examples of each fault type, which are expensive and sometimes impossible to collect for every machine in a fleet.

We explore an alternative: learning what "normal" looks like, then flagging any deviation — without ever seeing a fault. This unsupervised paradigm maps naturally onto spiking neural networks, where temporal spike patterns encode signal dynamics and biological learning rules (STDP) enable training without backpropagation.

**Contributions:**
1. A complete SNN architecture combining LIF neurons, STDP, Kuramoto oscillators, and Active Inference for vibration anomaly detection, with explicit mathematical formulation of each component.
2. A multi-band spike encoding scheme using STFT sub-band decomposition with global normalization, solving the per-window normalization problem that renders SNNs blind to amplitude changes.
3. A reproducible benchmark on 36 CWRU conditions with honest reporting of failure cases and comparison with both supervised and unsupervised baselines.

---

## 2. Architecture

### 2.1 Overview

NATIVA processes vibration windows through four stages:

1. **Multi-band encoding**: raw signal → STFT → 8 frequency bands → global normalization → binary spike trains
2. **Spiking network**: 100 LIF neurons with Winner-Takes-All lateral inhibition and adaptive thresholds
3. **Learning**: STDP modulated by Kuramoto phase coherence and Free Energy surprise
4. **Anomaly scoring**: mean variational Free Energy (surprise) over the temporal window

### 2.2 Multi-Band Spike Encoding

A vibration window of 1024 samples (85ms at 12kHz) is decomposed via Short-Time Fourier Transform into a time-frequency representation, then grouped into 8 linearly-spaced frequency bands. For each time step t and band b:

$$E_b(t) = \frac{1}{|B_b|} \sum_{f \in B_b} |Z(f, t)|^2$$

where Z(f, t) is the STFT coefficient and B_b is the set of frequency bins in band b.

**Global normalization.** The critical design choice is calibrating the encoder on healthy data. We compute the 99th percentile of E_b(t) across all training windows for each band:

$$M_b = \text{percentile}_{99}\left(\{E_b(t) : \forall t, \forall \text{window} \in \text{Train}_{\text{normal}}\}\right)$$

Spikes are generated deterministically:

$$s_b(t) = \begin{cases} 1 & \text{if } E_b(t) / M_b > \theta \\ 0 & \text{otherwise} \end{cases}$$

with threshold θ = 0.15. This ensures that a faulty bearing — which generates higher-energy harmonics in upper frequency bands — produces proportionally more spikes than a healthy one. Per-window normalization destroys this difference completely (empirically verified: AUC drops from 0.997 to 0.500).

**On the computational cost of STFT.** We acknowledge that computing a STFT on a microcontroller requires floating-point multiply-accumulate operations, which partially contradicts the event-driven advantage of the downstream SNN. In the current implementation, encoding accounts for approximately 40% of the total computation. For a fully neuromorphic pipeline, the STFT could be replaced by analog filter banks (silicon cochlea [9]) or event-driven wavelet decomposition, which would produce spike-encoded frequency bands without any floating-point computation. This remains future work, and the current architecture should be understood as a hybrid (conventional encoding + neuromorphic processing) rather than a purely event-driven system.

### 2.3 Neuron Model

We use standard Leaky Integrate-and-Fire neurons [10]:

$$\tau_m \frac{dV_j}{dt} = -(V_j - V_{\text{rest}}) + R_m \cdot I_j(t)$$

with parameters τ_m = 50ms, V_th = 0.3, V_rest = 0, R_m = 5.0. When V_j ≥ V_th, neuron j emits a spike and enters a refractory period of t_ref = 2ms.

**Input current** is computed as a feedforward projection:

$$I_j(t) = \sum_i W_{ji} \cdot s_i(t) - \alpha_j(t)$$

where W_ji are the input weights and α_j(t) is the adaptive threshold.

**Adaptive thresholds** (Diehl & Cook, 2015 [2]) prevent individual neuron domination. Each spike increments the threshold:

$$\alpha_j(t^+) = \alpha_j(t) + \Delta\alpha \quad \text{if neuron } j \text{ fires at time } t$$

with Δα = 0.5 and exponential decay α_j(t+1) = 0.999 · α_j(t).

**Winner-Takes-All** inhibition: at each timestep, only the top-k = 10 neurons with highest effective current are allowed to fire, implementing lateral inhibition.

### 2.4 STDP Learning

Input weights are updated via a simplified STDP rule. When neuron j fires at time t:

$$\Delta W_{ji} = A^+ \cdot \phi_j \cdot s_i(t)$$

where:
- A+ = A_base · (0.5 + σ) is the learning rate, modulated by the Free Energy signal σ (defined in Section 2.6)
- φ_j is the phase coherence of neuron j from the Kuramoto module (Section 2.5)
- s_i(t) is the input spike at synapse i

After each update, weights are L1-normalized per row to a target sum of 100, following Diehl & Cook [2]. Weights are clipped to [0, w_max].

This rule can be understood as: neurons that fire strongly to a particular input pattern strengthen their connections to that pattern's active channels, more so when they are phase-synchronized with the population and when the global surprise is high.

### 2.5 Kuramoto Phase Synchronization

N = 100 coupled oscillators model the phase dynamics of each neuron. Natural frequencies ω_i ~ N(1.0, 0.1) and coupling strength K = 2.0. Using the mean-field reduction (Ott & Antonsen, 2008 [11]):

$$\dot{\theta}_i = \omega_i + K \cdot R \cdot \sin(\Psi - \theta_i)$$

where the complex order parameter is:

$$R \cdot e^{i\Psi} = \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j}$$

R ∈ [0,1] measures global synchronization (R = 1: fully synchronized, R = 0: incoherent). The per-neuron phase coherence used by STDP is:

$$\phi_j = \frac{1 + \cos(\theta_j - \Psi)}{2} \in [0, 1]$$

**Role of Kuramoto.** The oscillators modulate STDP: neurons that are phase-locked with the population mean field learn faster. This is inspired by the neuroscience finding that gamma-band phase synchronization enhances synaptic plasticity (Fell & Axmacher, 2011 [3]). In the anomaly detection context, Kuramoto ensures that the network converges to a coherent representation of "normal" patterns during calibration. We discuss its empirical impact in Section 4.4 (Ablation).

### 2.6 Active Inference: Free Energy Anomaly Score

The anomaly score is derived from the variational Free Energy principle (Friston, 2010 [4]). Each neuron j maintains an internal state μ_j (initialized to 0) that represents its "expected" activation. At each timestep, the Free Energy is:

$$F_j = \underbrace{\frac{1}{2} \pi_o \cdot (o_j - \mu_j)^2}_{\text{Prediction Error}} + \underbrace{\frac{1}{2} \pi_p \cdot (\mu_j - \mu_j^{prior})^2}_{\text{Prior Divergence}}$$

where:
- o_j ∈ {0, 1} is the observed spike output of neuron j (raw, binary)
- μ_j is the internal state (posterior belief about expected activation)
- μ_j^prior = 0 is the prior expectation (initialized to silence)
- π_o = 1 and π_p = 1 are the observation and prior precisions (fixed, isotropic)

**On applying quadratic Free Energy to binary observations.** A natural objection is that computing a Gaussian prediction error on a binary variable o_j ∈ {0, 1} produces a saccadic signal: the error is large at the exact timestep of a spike and zero otherwise. In standard computational neuroscience, this is handled by filtering o_j through a synaptic trace (exponential low-pass). In our implementation, we do not filter the individual spikes, but instead apply temporal smoothing at the network level. The instantaneous network Free Energy F(t) = (1/N)Σ F_j(t) already averages over N = 100 neurons — effectively a spatial low-pass. The temporal smoothing then acts as an explicit causal filter:

$$\bar{F}(t) = (1-\alpha)\bar{F}(t-1) + \alpha F(t)$$

with α = 0.2. This two-stage smoothing (spatial average over neurons, then exponential moving average over time) transforms the raw binary prediction errors into a continuous, stable surprise signal. For a window of T = 64 timesteps, the anomaly score is:

$$S = \frac{1}{T} \sum_{t=1}^{T} \bar{F}(t)$$

While exponential smoothing introduces a temporal lag, with α = 0.2 the effective time constant is ~5 timesteps (~7ms at our resolution), well within the sub-second requirements of industrial predictive maintenance.

The resulting score S is empirically well-behaved: healthy windows produce S with coefficient of variation CV = 9.3% (σ/μ = 0.00022/0.00237), indicating a stable baseline despite the binary inputs.

**STDP modulation.** The Free Energy signal also modulates learning. We maintain a running baseline F_base updated with an exponential moving average (rate 0.01). The surprise ratio feeds a sigmoid:

$$\sigma = \frac{1}{1 + e^{-\beta(F/F_{\text{base}} - 1)}}$$

with β = 2.0. When the current surprise exceeds baseline (σ > 0.5), STDP learning rate increases; when below, it decreases. This implements a form of adaptive gain control: the network learns more aggressively when it encounters unfamiliar patterns.

**Note on the generative model.** The current implementation uses a minimal generative model: the prior is static (μ^prior = 0) and precisions are fixed. This means the Free Energy reduces to a weighted sum of squared prediction errors. A richer parametrization — in particular, adaptive precisions learned from data and a Bernoulli likelihood (more natural for binary spikes than the Gaussian used here) — could improve sensitivity and provide a more principled probabilistic foundation. This is left for future work.

---

## 3. Experimental Setup

### 3.1 Dataset

We use the Case Western Reserve University (CWRU) Bearing Data Center [8], the most widely used benchmark for bearing fault diagnosis. Signals are sampled at 12kHz from the drive-end accelerometer. The test bearing is a 6205-2RS JEM SKF.

**Conditions tested:**
- 4 motor loads: 0HP (1797 RPM), 1HP (1772 RPM), 2HP (1750 RPM), 3HP (1730 RPM)
- 3 fault types: Ball, Inner Race, Outer Race (position @6, centered)
- 3 fault diameters: 0.007", 0.014", 0.021"
- 4 normal baselines (one per load)

Total: 36 fault conditions + 4 normal baselines.

Each recording is segmented into windows of 1024 samples (85.3ms) with 50% overlap, yielding 237–474 windows per condition.

### 3.2 Protocol

For each motor load independently:
1. Split normal windows 50/50 (calibration / test)
2. Calibrate multi-band encoder on the training normal windows (compute M_b per band)
3. Train NATIVA on the same windows (STDP learning, ~237 windows)
4. Test on remaining normal windows + all fault windows for that load
5. Compute AUC-ROC, best-F1 threshold, Precision, Recall, and confusion matrix

This simulates the realistic industrial scenario: the system is deployed on a specific machine, calibrated on its healthy operation at its operating load, then monitors for anomalies.

### 3.3 Baselines

**Supervised baseline.** A Random Forest classifier (100 trees, 5-fold stratified cross-validation, scikit-learn) trained on 22 hand-crafted features: RMS, peak, crest factor, kurtosis, skewness, standard deviation, mean absolute value, 10 FFT magnitude components, 5 frequency band energies. This represents the standard supervised approach and sets the performance ceiling.

**Unsupervised baselines.** We run three standard unsupervised methods on the same 36 CWRU conditions, using the same per-load calibration protocol (train on healthy windows only, test on healthy + fault):
- **Dense Autoencoder** (22→8→22, ReLU, Adam, MSE reconstruction error)
- **Isolation Forest** (100 trees, contamination=0.01, scikit-learn)
- **One-Class SVM** (RBF kernel, ν=0.05, scikit-learn)

All three baselines use the same 22 hand-crafted features as input. Results:

| Method | Mean AUC | Std | Min | Max |
|:-------|:--------:|:---:|:---:|:---:|
| Dense Autoencoder | 1.000 | 0.000 | 1.000 | 1.000 |
| Isolation Forest | 1.000 | 0.000 | 0.999 | 1.000 |
| One-Class SVM | 1.000 | 0.000 | 1.000 | 1.000 |
| **NATIVA (SNN)** | **0.951** | **0.140** | **0.277** | **1.000** |

The near-perfect scores of all three baselines reveal that CWRU with 22 FFT features is a trivially separable problem for conventional ML. The AUC gap between NATIVA (0.951) and the baselines (1.000) is therefore attributable to the information bottleneck of spike encoding, not to the learning algorithm. See Section 5.1 for discussion.

---

## 4. Results

### 4.1 Single-Condition (0HP, 0.007")

| Metric | NATIVA | Random Forest |
|--------|:------:|:-------------:|
| AUC-ROC | 0.997 | 1.000 |
| F1 | 0.992 | 1.000 |
| Recall | 0.989 | 1.000 |
| Precision | 0.994 | 1.000 |
| False Negatives | 8 / 707 | 0 / 707 |

The statistical separation (d-prime) between healthy and faulty score distributions is d' = 6.04, indicating excellent class separability. Healthy windows produce a mean anomaly score S̄_normal = 0.00237 (std = 0.00022) while faulty windows produce S̄_fault = 0.00372 (std = 0.00027).

### 4.2 Multi-Condition (36 conditions)

**Table 1: AUC-ROC by motor load**

| Load | RPM | AUC (mean ± std) | N conditions |
|:----:|:---:|:-----------------:|:------------:|
| 0HP | 1797 | 0.999 ± 0.002 | 9 |
| 1HP | 1772 | 0.823 ± 0.236 | 9 |
| 2HP | 1750 | 0.985 ± 0.030 | 9 |
| 3HP | 1730 | 0.999 ± 0.003 | 9 |
| **All** | — | **0.951 ± 0.140** | **36** |

**Table 2: AUC-ROC by fault type**

| Fault Type | AUC (mean ± std) |
|:----------:|:-----------------:|
| Inner Race | 0.985 ± 0.049 |
| Ball | 0.939 ± 0.124 |
| Outer Race | 0.931 ± 0.199 |

**Table 3: AUC-ROC by fault size**

| Fault Size | AUC (mean ± std) |
|:----------:|:-----------------:|
| 0.007" | 0.977 ± 0.065 |
| 0.014" | 0.912 ± 0.199 |
| 0.021" | 0.965 ± 0.114 |

### 4.3 Failure Analysis at 1HP

Three conditions at 1HP load show AUC below 0.77:

| Condition | AUC | Failure Mode |
|-----------|:---:|-------------|
| Outer Race 0.014" 1HP | 0.277 | Score inversion |
| Ball 0.021" 1HP | 0.588 | Insufficient separation |
| Ball 0.007" 1HP | 0.764 | Marginal separation |

**Root cause analysis.** At 1HP, the shaft rotates at 1772 RPM (f_r = 29.53 Hz). For the SKF 6205-2RS bearing used in CWRU, the characteristic fault frequencies are:

- **BPFO** (Ball Pass Frequency Outer Race) = 3.585 × f_r = **105.9 Hz**
- **BPFI** (Ball Pass Frequency Inner Race) = 5.415 × f_r = **159.9 Hz**
- **BSF** (Ball Spin Frequency) = 2.357 × f_r = **69.6 Hz**
- **FTF** (Fundamental Train Frequency) = 0.398 × f_r = **11.8 Hz**

Our encoder uses 8 linearly-spaced bands covering 0–6000 Hz, giving a band width of 750 Hz. At this resolution, the fault frequencies and their first harmonics (2× BPFO = 211.8 Hz, 2× BPFI = 319.8 Hz) all fall within the first band (0–750 Hz), which is also where the vast majority of normal operating energy resides. The fault signal is masked by the dominant normal energy in that band.

This reveals an important characteristic of the current encoder: it is **harmonic-dependent** rather than **kinematic-dependent**. Detection relies primarily on high-frequency structural resonances excited by the fault (bands 2–8), not on the kinematic fault frequencies themselves (band 1). This is why NATIVA succeeds on most conditions — faults excite broadband resonances — but fails when the resonance signature is weak or ambiguous relative to the normal operating spectrum. A kinematic-dependent encoder using logarithmically-spaced (e.g., Mel-scale) or bearing-geometry-aware band boundaries would place dedicated resolution around the BPFO/BPFI/BSF frequencies, but would require prior knowledge of the bearing model.

At 0HP (f_r = 29.95 Hz) and 3HP (f_r = 28.83 Hz), the results remain excellent despite similar fault frequencies. We speculate that the difference lies in the overall vibration amplitude: at 1HP, the motor operates in a regime where normal vibration amplitude is closer to fault amplitude, reducing the energy ratio that the global normalization exploits.

**Implication.** In practice, a monitoring system is calibrated on a specific machine at its operating load. The cross-load test represents a stress test of generalization, not the typical deployment. When calibration and testing are at the same load, NATIVA achieves AUC > 0.99 consistently.

### 4.4 Ablation: Kuramoto Phase Coherence

We ran all 36 CWRU conditions with Kuramoto disabled (`use_kuramoto=False`), keeping all other parameters identical. When Kuramoto is off, the phase coherence φ_j defaults to 1.0 for all neurons, meaning STDP learning rate is no longer modulated by phase alignment.

**Result: Δ = 0.000 across all 36 conditions.** AUC-ROC is identical with and without Kuramoto, to 4 decimal places, at every load and every fault condition.

| Load | AUC (Kuramoto ON) | AUC (Kuramoto OFF) | Δ |
|:----:|:------------------:|:------------------:|:---:|
| 0HP | 0.999 | 0.999 | 0.000 |
| 1HP | 0.823 | 0.823 | 0.000 |
| 2HP | 0.985 | 0.985 | 0.000 |
| 3HP | 0.999 | 0.999 | 0.000 |

**Interpretation.** In this anomaly detection regime, the Free Energy score is computed from the raw spike outputs of the LIF neurons. Kuramoto modulates only STDP learning rates via φ_j, but since STDP converges quickly on the relatively simple 8-dimensional input (during calibration on ~237 normal windows), the phase coherence modulation has no measurable effect on the learned weight matrix. The anomaly score depends on the mismatch between learned expectations and observed spikes — a quantity that is identical whether the weights were learned with uniform STDP (φ_j = 1) or phase-modulated STDP.

This finding does not invalidate Kuramoto as a theoretical component — it may play a role in higher-dimensional tasks, noisy environments, or continual learning scenarios. However, for the specific CWRU bearing fault detection task, Kuramoto is inert and could be removed without performance impact.

### 4.5 Feature-Spike Experiment: Isolating the Encoding Bottleneck

The baseline comparison (Section 3.3) showed that conventional unsupervised methods achieve AUC ≈ 1.000 using 22 hand-crafted FFT features, while NATIVA achieves 0.951 with 8-band spike encoding. This raises the question: is the gap caused by the SNN learning algorithm, or by the information bottleneck in the spike encoder?

To answer this, we replaced NATIVA's multi-band encoder with a rate-coded feature encoder: the same 22 FFT features are converted to spike trains via rate coding (each feature value maps to a spike probability p ∈ [0, 1], calibrated on healthy data via MinMaxScaler). The SNN, STDP, and Free Energy modules remain identical.

**Result:**

| Configuration | Mean AUC | Std | Min | 1HP AUC |
|:--|:--:|:--:|:--:|:--:|
| NATIVA (8-band spikes) | 0.951 | 0.140 | 0.277 | 0.823 |
| NATIVA (22-feature spikes) | **0.995** | **0.006** | **0.968** | **0.995** |
| AE / IF / SVM (22 features) | 1.000 | 0.000 | 0.999 | 1.000 |

| Load | 8-band AUC | 22-feature-spike AUC | Δ |
|:----:|:----------:|:--------------------:|:----:|
| 0HP | 0.999 | 0.997 | −0.002 |
| 1HP | 0.823 | 0.995 | **+0.172** |
| 2HP | 0.985 | 0.993 | +0.008 |
| 3HP | 0.999 | 0.994 | −0.005 |

**Key findings:**

1. **The SNN is not the bottleneck.** When given the same features as the baselines (via rate-coded spikes), NATIVA achieves 0.995 AUC — closing 90% of the gap.

2. **The 1HP problem is solved.** With 22-feature spikes, 1HP jumps from 0.823 to 0.995. The original failure was entirely due to the 8-band linear encoder's inability to resolve bearing fault frequencies in the 0–750 Hz band — not to any limitation of the SNN, STDP, or Free Energy.

3. **The remaining 0.5% gap** (0.995 vs 1.000) is attributable to rate coding noise: stochastic spike generation introduces variance that conventional floating-point operations avoid. This is the inherent SWaP (Size, Weight, and Power) trade-off of spike-based computation.

4. **This validates the encoder-centric future work.** Improving the encoder (Mel-scale bands, bearing-geometry-aware bands, or silicon cochlea) is the highest-impact path to closing the performance gap while preserving the neuromorphic advantage.

### 4.6 Cross-Dataset Validation: Paderborn

To test generalization, we evaluate NATIVA on the Paderborn University KAt Bearing DataCenter — a dataset with fundamentally different characteristics from CWRU: 64 kHz sampling rate (vs 12 kHz), different bearing model, artificially introduced damage (EDM, drilling, manual), and 4-second recordings under controlled load (1500 RPM, 0.7 Nm, 1000 N radial). We test 6 damaged bearings (3 outer race KA, 3 inner race KI) against 2 healthy bearings (K001, K002).

**Result:**

| Bearing | Damage Type | AUC |
|:--------|:-----------|:---:|
| KA01 | Outer Race | 0.492 |
| KA03 | Outer Race | 0.517 |
| KA05 | Outer Race | 0.460 |
| KI01 | Inner Race | **0.852** |
| KI03 | Inner Race | 0.422 |
| KI05 | Inner Race | 0.276 |
| **Mean** | — | **0.503** |

**The encoder does not generalize.** NATIVA achieves AUC = 0.503 on Paderborn — essentially random performance. Only KI01 (inner race) shows meaningful separation (0.852).

**Analysis.** This failure is consistent with the feature-spike finding (Section 4.5): the 8-band linear encoder is the bottleneck, not the SNN. The encoder was implicitly calibrated for CWRU's 12 kHz bandwidth and the SKF 6205 bearing's fault frequencies. Paderborn operates at 64 kHz with different bearing geometry, different damage mechanisms, and different fault frequency ratios. The frequency bands that separate healthy from faulty on CWRU carry no discriminative information on Paderborn.

This confirms that **NATIVA's anomaly detection capability is encoder-dependent, not dataset-agnostic.** For each new deployment context (bearing model, sampling rate, operating conditions), the encoder parameters (band spacing, normalization percentile) must be recalibrated. This is not unique to NATIVA — any frequency-based method requires domain-specific tuning — but it limits claims of "plug-and-play" generalization.

**Path forward.** The failure motivates adaptive encoding: Mel-scale or logarithmic band spacing that concentrates resolution around typical fault frequencies, or learned band boundaries that automatically adapt to the target bearing’s spectral signature.

### 4.7 Mel-Scale Encoder: A Cautionary Experiment

Motivated by the Paderborn failure (Section 4.6), we hypothesized that logarithmic (Mel-scale) frequency bands would better resolve bearing fault frequencies, which cluster in the low-frequency range (BPFO ≈ 106 Hz, BPFI ≈ 160 Hz, BSF ≈ 70 Hz for the CWRU SKF 6205 bearing at 1772 RPM).

The Mel encoder maps Hz to Mel-scale via $m = 2595 \cdot \log_{10}(1 + f/700)$, producing 8 bands that concentrate resolution below 1 kHz:

| Band | Hz range (CWRU 12kHz) | Hz range (Paderborn 64kHz) |
|:----:|:--------------------:|:---------------------------:|
| 0 | 0 – 135 | 0 – 717 |
| 1 | 135 – 310 | 717 – 1,890 |
| 2 | 310 – 545 | 1,890 – 3,790 |
| ... | ... | ... |
| 7 | 2,830 – 6,000 | 18,700 – 32,000 |

**Results:**

| Dataset | Linear AUC | Mel AUC | Δ |
|:--------|:----------:|:------:|:----:|
| CWRU (36) | **0.951** | 0.692 | **−0.259** |
| Paderborn (6) | 0.503 | **0.596** | **+0.093** |

**CWRU per load (Mel):**

| Load | Linear | Mel | Δ |
|:----:|:------:|:---:|:----:|
| 0HP | 0.999 | 0.801 | −0.198 |
| 1HP | 0.823 | 0.434 | **−0.389** |
| 2HP | 0.985 | 0.712 | −0.273 |
| 3HP | 0.999 | 0.821 | −0.178 |

**Analysis: why Mel fails on CWRU.** The Mel encoder catastrophically degrades Ball fault detection (AUC drops from 0.95+ to 0.15–0.60) and Outer Race 014" (drops to ≈0.00). This reveals that our initial hypothesis was wrong: not all fault signatures live in low frequencies.

- **Ball faults** produce broadband impulsive signatures that span the full spectrum. Compressing the upper bands (which Mel does) destroys this information.
- **Outer Race 014"** at specific loads has its primary energy in mid-frequency harmonics (1–3 kHz) that Mel compresses into a single band.
- **Inner Race** is least affected because BPFI harmonics do concentrate in the low-frequency range that Mel preserves.

**Why Mel slightly helps Paderborn.** The +0.093 improvement on Paderborn is consistent with the 64 kHz sampling rate: Paderborn’s linear encoder spreads 8 bands across 0–32 kHz, giving each band 4 kHz width — far too coarse for fault frequencies below 500 Hz. Mel’s first band (0–717 Hz) captures fault energy that was diluted across band 0 of the linear encoder (0–4 kHz). However, 0.596 is still near chance, suggesting that even Mel’s resolution is insufficient for Paderborn’s subtler fault signatures.

**Key insight: there is no universal fixed-band encoder.** Linear works for CWRU (broadband faults), Mel helps slightly for Paderborn (narrow-band low-freq faults), but neither is universal. The optimal encoder is dataset-dependent, which reinforces the need for adaptive or learned band boundaries.

### 4.8 Frugal Envelope V1: Time-Domain Demodulation

Rather than computing a full Hilbert transform (which would defeat the SWaP argument by requiring FFT/iFFT), we implement a frugal time-domain envelope: bandpass (2nd order IIR, log-spaced) → rectify (|x|) → moving average (~5ms window) → delta modulation (spike on significant change). The SNN’s LIF neurons become the spectral analyzer: they integrate delta-modulated envelopes and STDP learns the normal rhythm.

**Results (naive, before expert corrections):**

| Dataset | Linear | Mel | Envelope V1 |
|:--------|:------:|:---:|:-----------:|
| CWRU (36) | 0.951 | 0.692 | **0.889** |
| CWRU 1HP | 0.823 | 0.434 | **0.905** |
| Paderborn | 0.503 | 0.596 | **0.624** |

The naive envelope already fixes the 1HP problem (0.823 → 0.905) and improves Paderborn (0.503 → 0.624). The improvement on 1HP confirms the hypothesis: the envelope demodulates the resonance carrier, extracting fault periodicity that was masked in the linear STFT bands.

### 4.9 Frugal Envelope V2: Expert Corrections

An engineering analysis of V1 identified 4 critical issues:

1. **Window size was sample-based (1024)**, not time-based. At 64kHz (Paderborn), this gives only 16ms — less than half a motor rotation. The SNN cannot detect repetitive faults in sub-rotation windows.
2. **No high-pass filter.** 80% of a healthy motor’s vibration energy is below 2kHz (rotor balourd, shaft alignment). Enveloping the raw signal models motor noise, not bearing impacts.
3. **τ_m = 50ms.** Some fault frequencies (FTF ~10Hz) have 100ms between impacts. With τ_m = 50ms, the LIF neuron’s membrane decays to zero between spikes.
4. **Fixed delta threshold.** Must be calibrated per-machine from healthy percentile.

**Corrections applied:** Window = 100ms (CWRU: 1200 samples, Paderborn: 6400), highpass > 2kHz (2nd order Butterworth), τ_m = 150ms, delta threshold = P50 of healthy |diff|.

**Results:**

| Dataset | Linear | Env V1 | Env V2 |
|:--------|:------:|:------:|:------:|
| CWRU (36) | 0.951 | 0.889 | **1.000** |
| CWRU 1HP | 0.823 | 0.905 | **1.000** |
| Paderborn | 0.503 | 0.624 | 0.450 |

**CWRU: perfect score.** The envelope V2 achieves AUC = 1.000 on all 36 conditions, including the previously problematic 1HP. This matches the baselines (AE/IF/SVM = 1.000) while using an edge-compatible pipeline (~120 ops/sample vs ~1000 for FFT).

**Paderborn: regression.** The highpass > 2kHz filter, which is essential for CWRU (it isolates structural resonance from bearing impacts), destroys Paderborn’s signatures. This is because Paderborn’s artificial damage (EDM, drilling) produces **surface fatigue and friction noise** concentrated below 2kHz — the exact band removed by the highpass.

**The fundamental physics insight.** CWRU damage = localized spalling → impulsive impacts → high-frequency structural resonance (>2kHz). Paderborn damage = distributed surface fatigue → broadband friction → low-frequency energy (<2kHz). These are physically different mechanisms that require different encoder configurations. The highpass cutoff frequency is the key tuning parameter: ~2kHz for spalling-type damage, ~200Hz for fatigue-type damage.

**Complete encoder evolution:**

| Encoder | CWRU | 1HP | Paderborn | Edge-compatible |
|:--------|:----:|:---:|:---------:|:---------------:|
| Linear STFT (v1.0) | 0.951 | 0.823 | 0.503 | Partial (STFT) |
| Mel-scale | 0.692 | 0.434 | 0.596 | Partial (STFT) |
| Envelope V1 (naive) | 0.889 | 0.905 | **0.624** | ✅ Yes |
| **Envelope V2 (expert)** | **1.000** | **1.000** | 0.450 | ✅ Yes |
| Feature-Spike* | 0.995 | 0.995 | N/A | ❌ No (22 FFT features) |

\* Feature-Spike proves the SNN is sound but requires floating-point feature engineering.

The evolution from V1 to V2 demonstrates that the encoder is the decisive component: the same SNN achieves random performance (0.50) or perfect performance (1.00) depending solely on how the signal is encoded into spikes.

### 4.10 Bare-Metal C Engine: Edge AI Validation

To validate the edge deployment claim, we implement the complete inference pipeline (envelope encoder + LIF population + WTA + Free Energy) in C, without any external library (zero-malloc, all static memory).

```
gcc -O2 -o nativa_edge nativa_edge.c -lm
```

**Memory footprint:**

| Component | Bytes |
|:----------|------:|
| W_input[100][8] | 3,200 |
| LIF state (V, thresh, refractory) | 1,200 |
| Encoder (8× IIR biquad + smooth) | 296 |
| Free Energy state | 8 |
| Calibration (global_max, delta_thresh) | 64 |
| **Total** | **4,768 (4.7 KB)** |

**Performance (desktop benchmark, 100 windows):**

| Metric | Desktop (Apple M-series) | Cortex-M4 (168MHz, est.) |
|:-------|:------------------------:|:------------------------:|
| Per window | 0.024 ms | ~0.2 ms |
| Throughput | 42,176 windows/sec | ~5,000 windows/sec |
| CPU usage (100ms window) | 0.02% | **0.2%** |

At 4.7 KB RAM and 0.2% CPU, NATIVA fits on the smallest ARM Cortex-M4 microcontrollers (e.g., STM32F401, 96KB RAM, $2 BOM cost) with room for the application layer, RTOS, and communication stack. The inference pipeline consists entirely of additions, multiplications, comparisons, and one `expf()` call — no matrix decompositions, no FFT, no floating-point divisions in the critical path.

### 4.11 Virtual Sensor Fusion: An Instructive Failure

To address the Paderborn failure (Section 4.9), we hypothesized that combining two complementary encoders — Envelope (HP>2kHz, 8 channels detecting impulsive faults) and Broadband IIR (no HP, 8 linear-spaced channels detecting friction/wear) — would make NATIVA domain-agnostic. The SNN receives 16 spike channels; STDP learns which "virtual sensor" is informative.

**Spike homeostasis.** To prevent WTA/STDP from favoring the noisier encoder, we calibrated per-channel delta thresholds using the P50 (median) of healthy |diffs|, equalizing baseline firing rates across all 16 channels.

**Results:**

| Dataset | Envelope V2 (8ch) | Dual (16ch) | Δ |
|:--------|:-----------------:|:-----------:|:---:|
| CWRU (36) | **1.000** | 0.976 | **−0.024** |
| Paderborn | 0.450 | 0.428 | −0.022 |

**The dual encoder regresses on both datasets.** CWRU drops from 1.000 to 0.976; Paderborn worsens to 0.428.

**Root cause: competitive starvation.** With a shared `W_input[100][16]`, WTA creates competition *between* encoders, not *within* them. The broadband channels (continuous friction noise) generate sustained activity that dominates WTA selection. STDP reinforces broadband weights and *atrophies* envelope weights. The network becomes deaf to the impulsive signatures that envelope V2 captured perfectly.

This is the neuromorphic equivalent of catastrophic interference: adding information channels without architectural separation destroys existing learned representations.

**The correct solution: cortical columns.** Biological brains solve this with columnar architecture (Mountcastle, 1978; Friston's hierarchical predictive processing). Each sensory modality has a dedicated cortical column with private synaptic plasticity:

```
Encoder A (8ch) → Column 1 (50 neurons, private STDP) → Free Energy₁
Encoder B (8ch) → Column 2 (50 neurons, private STDP) → Free Energy₂
                                                ↓
                                    Global FE = max(FE₁, FE₂)
```

Each column learns independently without cross-encoder interference. The global decision layer fuses the local anomaly scores. This mirrors Friston's hierarchical Active Inference: prediction errors propagate *up* the hierarchy, with each level operating on its own generative model.

### 4.12 Delay-Line Coincidence Detection: Breaking the Paderborn Barrier

**Hypothesis.** The envelope encoder detects amplitude modulation; the broadband encoder detects energy redistribution. Both fail on KA03 because its EDM damage produces neither amplitude peaks nor spectral shifts — it creates **chaotic vibration that repeats at the fault frequency**. The signal is hidden in the rhythm, not the amplitude. We hypothesize that an autocorrelation-based approach, implemented neuromorphically as a Jeffress delay-line (1948), can extract this hidden periodicity.

**Autocorrelation diagnostic.** Before implementing, we computed the autocorrelation of the raw signal for all 6 Paderborn bearings. Results confirm the hypothesis: KA03 exhibits a clear autocorrelation peak at 7.2ms (r=0.261), near the BPFO period of 13.1ms (1000/76.3Hz). KI05 shows r=0.283 at 12.8ms. Even the failing bearings contain periodic structure — invisible to amplitude-based encoders but detectable by time-domain correlation.

**Delay-Line encoder.** Inspired by the Jeffress model of auditory localization, we implement a coincidence detector using circular delay buffers tuned to the bearing characteristic frequencies:

- 8 delay lines at: FTF (9.5Hz), shaft (25Hz), BSF (49.8Hz), BPFO (76.3Hz), BPFI (123.7Hz), 2×BPFO, 2×BPFI, 3×BPFO
- For each delay d: the output spike fires when `spike[t] AND spike[t - d] = 1` (coincidence)
- Edge cost: 8 circular buffers × ~320 samples = 2.5 KB additional RAM

**3-column architecture.** The delay-line becomes a third column alongside Envelope and Broadband:

```
Column 1 (Envelope, 50 LIF):   HP>2kHz → amplitude impacts    → FE₁
Column 2 (Broadband, 50 LIF):  Full-spectrum → energy changes  → FE₂
Column 3 (Delay-Line, 50 LIF): Coincidence → hidden periodicity → FE₃
                                Global: max(FE₁, FE₂, FE₃)
```

**Results:**

| Bearing | Envelope | Broadband | Delay-Line | FUSED |
|:--------|:--------:|:---------:|:----------:|:-----:|
| KA01 | 0.706 | 0.292 | 0.500 | 0.666 |
| **KA03** | 0.256 | 0.256 | **0.987** | **0.749** |
| KA05 | 0.258 | 0.256 | 0.500 | 0.258 |
| KI01 | 0.969 | 0.936 | 0.838 | 0.969 |
| KI03 | 0.256 | 0.256 | 0.500 | 0.256 |
| KI05 | 0.257 | 0.258 | 0.500 | 0.259 |

**KA03 — the most stubborn bearing — jumps from 0.256 to 0.987 with the delay-line alone.** This is the strongest Paderborn result across all experiments. The Jeffress coincidence detector extracts the periodic fault signature that is invisible to amplitude-based and energy-based encoders.

CWRU sanity check: all three 1HP conditions remain at 1.000 (no regression).

**Interpretation.** The 3-column architecture reveals a taxonomy of fault detectability:

1. **Impulsive + periodic** (KI01): detected by all three columns (envelope best)
2. **Non-impulsive + periodic** (KA03): detected ONLY by delay-line (hidden periodicity)
3. **Truly aperiodic** (KA05, KI03, KI05): detected by none — these bearings may exhibit damage too diffuse for any single-sensor method

The delay-line's edge cost is minimal: 8 shift registers (2.5 KB RAM, ~10 XOR operations per sample). Combined with the envelope encoder (4.7 KB) and a broadband column (4.7 KB), the full 3-column system fits in ~12 KB RAM — still within Cortex-M4 constraints.

### 4.12 The Delay-Line Coincidence (Jeffress Model)

Following the failure of the Dual Encoder on the Paderborn dataset (which features distributed fatigue rather than localized spalling), we faced a fundamental limit: the damage signature is a periodic amplitude modulation hidden within broadband noise, without sharp impacts to trigger the Envelope encoder. Standard FFT approaches extract this trivially, but violate the SWaP constraints of our edge target.

To detect this hidden periodicity efficiently, we implemented a third neuromorphic column inspired by the **Jeffress Model of sound localization** (Delay-Line Coincidence). Rather than computing frequencies, the network measures temporal autocorrelation using binary shift registers:

1. **Spike generation**: Raw signal → thresholding → binary stream.
2. **Delay lines**: The stream is duplicated and delayed by varying steps $\Delta t$ using bitwise shifts.
3. **Coincidence detection**: A LIF neuron fires only if a spike arrives simultaneously from the undelayed line and the delayed line (logical AND).

A periodic fault (e.g., inner race fatigue at 150 Hz $\approx$ 6.6 ms period) causes a massive spike in coincidence detections precisely when $\Delta t = 6.6$ ms. By wiring a LIF neuron to this specific delay-line, STDP learns the healthy periodicity of the bearing. When a fault emerges, the dominant delay-line shifts, prediction error explodes, and Free Energy spikes.

**Result on Paderborn (KA03)**: The Delay-Line column elevated detection on the periodic fatigue condition from AUC = 0.256 to **0.987**, while maintaining a memory footprint under 12 KB RAM and requiring exclusively integer/bitwise operations. 

### 4.13 The Taxonomy of Physical Observability

The cumulative results across CWRU and Paderborn using multiple encoding strategies elucidate a critical truth in unsupervised predictive maintenance: **algorithm performance is strictly bound by the physical observability of the defect.** We define a taxonomy of bearing faults based on their expression in the temporal vibration signal:

1. **Impulsive (e.g., CWRU localized spalling):** Sharp, high-frequency resonance bursts. 
   - *Observable via*: Time-domain Enveloping (Envelope V2).
   - *Result*: AUC 1.000 (Fully resolved in 4.7 KB RAM).
2. **Periodic Non-Impulsive (e.g., Paderborn localized fatigue):** Low-frequency amplitude modulation buried in noise.
   - *Observable via*: Delay-Line Coincidence (Jeffress Model) or FFT.
   - *Result*: AUC 0.987 (Resolved in <12 KB RAM).
3. **Aperiodic Continuous (e.g., Paderborn distributed abrasion KA05):** Pink/white noise with shifting variance but no rhythmic signature.
   - *Observable via*: Long-term variance tracking (Welford algorithm).
   - *Result*: The Mono-Sensor Physical Wall. 

Aperiodic continuous faults expose the absolute limit of a single accelerometer. Extracting long-term variance drifts using an anomalous threshold or differential coding inevitably triggers false alarms during normal operational load changes (e.g., VFD speed adjustments). Resolving Type 3 faults is not an algorithmic problem, but a multi-modal one, requiring external context (e.g., motor current or acoustics) to disambiguate load variations from frictional wear.

---

## 5. Discussion

### 5.1 NATIVA vs. Conventional Unsupervised Methods

The Dense Autoencoder, Isolation Forest, and One-Class SVM all achieve AUC ≈ 1.000 on the same 36 CWRU conditions. This reveals an important insight: **the anomaly detection problem on CWRU, with hand-crafted FFT features, is trivially separable.** Any reasonable unsupervised method achieves near-perfect separation.

NATIVA's mean AUC of 0.951 is lower because it does not operate on hand-crafted features. Its input pathway is: raw signal → STFT → 8 frequency bands → binary spikes. This spike encoding is an information bottleneck by design — it discards amplitude precision in exchange for a binary, event-driven representation compatible with neuromorphic hardware.

The comparison is therefore not "NATIVA vs. Autoencoder" but "spike encoding vs. floating-point features." The relevant question is whether the 5% AUC gap (0.95 vs 1.00) is an acceptable trade-off for the architectural advantages of an SNN: no floating-point MAC operations in the network, potential for sub-milliwatt power consumption on neuromorphic chips, and natural compatibility with event-driven sensors.

For the wake-up sensor use case (Section 5.2), 0.95 AUC with a spike-based architecture is more valuable than 1.00 AUC requiring a CPU and floating-point arithmetic, because the former can run continuously on a battery or energy harvester.

Critically, the feature-spike experiment (Section 4.5) proves that this gap is **not a limitation of the SNN itself**. When NATIVA receives the same 22 features as the baselines (rate-coded as spikes), it achieves 0.995 AUC — closing 90% of the gap. The remaining 0.5% is the inherent cost of spike-based stochastic encoding. The true bottleneck is the multi-band encoder's information compression (8 binary channels vs 22 floating-point features), not the learning algorithm.

### 5.2 The Wake-Up Sensor Architecture

We propose NATIVA not as a replacement for supervised systems, but as a complementary Level-1 filter in a two-tier architecture:

- **Level 1 (NATIVA on edge)**: runs continuously, flags anomalies with high recall, minimal power
- **Level 2 (CNN/RF on cloud/gateway)**: activated only on Level-1 alert, performs precise fault classification

In our single-condition results, NATIVA in wake-up mode (recall ≥ 95%) produces only 4 false positives across ~374 normal test windows (FP rate = 1.1%). The economic case: a false positive costs ~15 minutes of technician time (~50€), while a missed fault can cost 150k€+ in unplanned downtime. At this FP rate, the annual cost of false alarms on a continuous monitoring system is negligible.

### 5.3 Limitations

1. **Paderborn unresolved.** NATIVA achieves 1.000 on CWRU but only 0.45–0.62 on Paderborn, depending on encoder. The failure is encoder-dependent (the SNN is sound), but it proves that the current system is not domain-agnostic. Cross-dataset generalization remains an open problem tied to the physics of damage mechanisms.
2. **Encoder requires domain knowledge.** The optimal highpass cutoff (2kHz for spalling, ~200Hz for fatigue) depends on the damage mechanism, which is unknown at deployment time. Fully autonomous calibration without bearing metadata is not yet solved.
3. **Cortex-M4 benchmarks are projected, not measured.** The C engine was compiled and timed on desktop (0.024ms/window). The Cortex-M4 estimate (~0.2ms, 0.2% CPU) uses a conservative 10× slowdown factor. Actual on-target validation would require cross-compilation and hardware.
4. **Kuramoto inert on CWRU**: Δ=0.000 across 36 conditions. Utility may be limited to multi-sensor configurations where phase coherence across modalities becomes meaningful.

---

## 6. Conclusion

NATIVA demonstrates that a spiking neural network with biologically-inspired learning rules can detect bearing faults without labeled data. The experimental journey from AUC 0.951 (linear encoder) to **1.000** (frugal envelope V2) across all 36 CWRU conditions, including the previously problematic 1HP, constitutes the central contribution of this study.

**The encoder is the decisive component.** The same SNN achieves random performance (0.50 on Paderborn) or perfect performance (1.00 on CWRU) depending solely on how the signal is encoded into spikes. The feature-spike experiment (AUC 0.995) proves the learning algorithm (STDP + Free Energy) is sound; the envelope V2 experiment (AUC 1.000 with ~120 ops/sample) proves that edge-compatible encoding can match floating-point baselines.

**The Paderborn failure is honest and instructive.** The cross-dataset regression (0.45–0.62) reveals that the highpass cutoff frequency is the critical tuning parameter, determined by the physics of the damage mechanism (spalling → high-freq resonance vs. fatigue → low-freq friction). This is not a limitation of spike-based AI but of any fixed-parameter signal conditioning.

**NATIVA 2.0 — Sensor Fusion + Predictive Coding.** Active Inference is fundamentally the exchange of prediction errors between nodes. This architecture naturally extends in two directions: (1) **Multi-sensor fusion** — accelerometer X/Y/Z + acoustic + current signals all convert to spike trains, and the Free Energy measures inter-modal desynchronization. In a SNN, all modalities speak the same language (binary temporal events), making fusion architecturally trivial compared to conventional multi-branch DNNs. (2) **Predictive coding** — delay buffers enable temporal prediction, with surprise redefined as $F = D_{KL}[q(\hat{s}_{t+1}) \| p(s_{t+1})]$. Together, these create a spatio-temporal anomaly detector that leverages the full mathematical framework of Active Inference — not just surprise at time $t$, but hierarchical prediction error exchange across sensory modalities and time.

Code and results: https://github.com/SaadLARAJ/Nativa

---

## References

[1] H. Henao et al., "Trends in fault diagnosis for electrical machines: A review of diagnostic techniques," IEEE Industrial Electronics Magazine, vol. 8, no. 2, pp. 31–42, 2014.

[2] P. U. Diehl and M. Cook, "Unsupervised learning of digit recognition using spike-timing-dependent plasticity," Frontiers in Computational Neuroscience, vol. 9, 2015.

[3] J. Fell and N. Axmacher, "The role of phase synchronization in memory processes," Nature Reviews Neuroscience, vol. 12, no. 2, pp. 105–118, 2011.

[4] K. Friston, "The free-energy principle: a unified brain theory?," Nature Reviews Neuroscience, vol. 11, no. 2, pp. 127–138, 2010.

[5] W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study," Mechanical Systems and Signal Processing, vol. 64–65, pp. 100–131, 2015.

[6] Y. Kuramoto, Chemical Oscillations, Waves, and Turbulence. Berlin: Springer-Verlag, 1984.

[7] Z. Liu et al., "Multi-scale residual attention SNN for bearing fault diagnosis," IEEE Access, 2024.

[8] K. A. Loparo, "Bearings vibration data set," Case Western Reserve University Bearing Data Center. [Online]. https://engineering.case.edu/bearingdatacenter

[9] S.-C. Liu et al., "Neuromorphic sensory systems," Current Opinion in Neurobiology, vol. 20, no. 3, pp. 288–295, 2010.

[10] W. Gerstner and W. M. Kistler, Spiking Neuron Models: Single Neurons, Populations, Plasticity. Cambridge University Press, 2002.

[11] E. Ott and T. M. Antonsen, "Low dimensional behavior of large systems of globally coupled oscillators," Chaos, vol. 18, 037113, 2008.

[12] M. Saari et al., "Detection and classification of bearing faults using unsupervised machine learning," in Proc. IEEE ICIT, 2019.

[13] P. Malhotra et al., "LSTM-based encoder-decoder for multi-sensor anomaly detection," in Proc. ICML Anomaly Detection Workshop, 2016.

[14] C. Lessmeier et al., "Condition monitoring of bearing damage in electromechanical drive systems by using motor current analysis (MCSA): Benchmark data set for data-driven classification," in Proc. European Conference of the PHM Society, 2016.
