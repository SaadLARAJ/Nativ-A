"""
NATIVA — Feature-Spike Experiment
====================================
Key question: Is the AUC gap (0.95 vs 1.00) caused by the SNN learning,
or by the spike encoding bottleneck?

Method:
    Instead of STFT → 8 bands → binary spikes, we:
    1. Extract the same 22 FFT features that gave AE/IF/SVM their 1.000 AUC
    2. Convert them to rate-coded spike trains (continuous → spike probability)
    3. Feed them to NATIVA (same network, same STDP, same Free Energy)

    If NATIVA achieves ~1.000 with these features, the SNN is proven good —
    the 8-band encoder was the bottleneck, not the learning algorithm.

Output:
    - results/feature_spike_results.json
    - results/feature_spike_report.png
"""

import numpy as np
import os
import sys
import json
import io
import time

from scipy.io import loadmat
from scipy.signal import stft
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- NATIVA import ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nativa'))
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from nativa_network import NativaNetwork, NativaConfig
    from neuron import LIFNeuron
finally:
    sys.stdout = _real_stdout


# =====================================================================
# CWRU CATALOG
# =====================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cwru')
SAMPLE_RATE = 12000

NORMAL_FILES = {
    0: {'id': 97,  'rpm': 1797},
    1: {'id': 98,  'rpm': 1772},
    2: {'id': 99,  'rpm': 1750},
    3: {'id': 100, 'rpm': 1730},
}

FAULT_FILES = {
    'Ball':  {'007': {0:118,1:119,2:120,3:121}, '014': {0:185,1:186,2:187,3:188}, '021': {0:222,1:223,2:224,3:225}},
    'Inner': {'007': {0:105,1:106,2:107,3:108}, '014': {0:169,1:170,2:171,3:172}, '021': {0:209,1:210,2:211,3:212}},
    'Outer': {'007': {0:130,1:131,2:132,3:133}, '014': {0:197,1:198,2:199,3:200}, '021': {0:234,1:235,2:236,3:237}},
}


def _load_signal(file_id):
    fpath = os.path.join(DATA_DIR, f"{file_id}.mat")
    if not os.path.exists(fpath):
        return None
    mat = loadmat(fpath)
    key = f"X{file_id:03d}_DE_time"
    if key not in mat:
        key2 = f"X{file_id}_DE_time"
        if key2 in mat: key = key2
        else:
            candidates = [k for k in mat if 'DE_time' in k]
            if not candidates: return None
            key = candidates[0]
    return mat[key].flatten()


def _segment(signal, window_size=1024, overlap=0.5):
    step = int(window_size * (1 - overlap))
    n = (len(signal) - window_size) // step
    return np.array([signal[i*step : i*step + window_size] for i in range(n)])


# =====================================================================
# FEATURE EXTRACTION (same 22 features as baseline_autoencoder.py)
# =====================================================================

def extract_features(window):
    """22 hand-crafted features: 7 time + 10 FFT + 5 band energies."""
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(np.abs(window))
    crest = peak / (rms + 1e-10)
    kurt = float(np.mean((window - np.mean(window))**4) / (np.std(window)**4 + 1e-10))
    skew = float(np.mean((window - np.mean(window))**3) / (np.std(window)**3 + 1e-10))
    std = np.std(window)
    mae = np.mean(np.abs(window))

    fft_mag = np.abs(np.fft.rfft(window))
    fft_mag = fft_mag / (np.max(fft_mag) + 1e-10)
    n_fft = len(fft_mag)
    indices = np.linspace(0, n_fft - 1, 10, dtype=int)
    fft_features = fft_mag[indices]

    n_bands = 5
    band_size = n_fft // n_bands
    band_energies = np.array([
        np.mean(fft_mag[i*band_size:(i+1)*band_size]**2) for i in range(n_bands)
    ])

    return np.concatenate([[rms, peak, crest, kurt, skew, std, mae],
                           fft_features, band_energies])


# =====================================================================
# RATE-CODED SPIKE ENCODER FOR FEATURES
# =====================================================================

class FeatureSpikeEncoder:
    """
    Converts 22 continuous features into spike trains via rate coding.

    Each feature value is mapped to a spike probability p ∈ [0, 1].
    At each timestep, a spike is emitted with probability p.

    Calibrated on healthy data (MinMaxScaler to [0, 1]).
    """
    def __init__(self, n_features=22, n_time_steps=64, seed=42):
        self.n_features = n_features
        self.n_time_steps = n_time_steps
        self.scaler = MinMaxScaler(clip=True)  # clip to [0, 1]
        self.rng = np.random.RandomState(seed)

    def calibrate(self, feature_matrix):
        """Fit scaler on healthy feature vectors."""
        self.scaler.fit(feature_matrix)

    def encode(self, feature_vector):
        """Convert 22 features → 64×22 binary spike matrix."""
        # Scale to [0, 1]
        scaled = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
        # Clamp probabilities
        probs = np.clip(scaled, 0.01, 0.99)
        # Generate spike train: each timestep, spike with probability p
        spikes = np.zeros((self.n_time_steps, self.n_features))
        for t in range(self.n_time_steps):
            spikes[t, :] = (self.rng.random(self.n_features) < probs).astype(np.float64)
        return spikes


# =====================================================================
# RUN FEATURE-SPIKE NATIVA
# =====================================================================

def run_feature_spike(normal_windows, fault_windows, condition_name):
    """
    Run NATIVA with 22 FFT features rate-coded as spikes.
    Returns AUC.
    """
    # Config — 22 inputs instead of 8
    params = LIFNeuron(tau_m=50.0, V_th=0.3, R_m=5.0, t_ref=2.0)
    cfg = NativaConfig(
        n_neurons=100, n_output_classes=2, neuron_params=params,
        use_kuramoto=True, kuramoto_coupling=2.0,
        weight_norm_target=100.0, use_adaptive_thresh=True,
        thresh_increment=0.5, seed=42
    )

    # Override input size to 22
    cfg_dict = cfg.__dict__.copy()
    net = NativaNetwork(cfg)
    # The network auto-adapts input dimension from the first spike matrix

    encoder = FeatureSpikeEncoder(n_features=22, n_time_steps=64)

    # Split 50/50
    np.random.seed(42)
    n_train = len(normal_windows) // 2
    perm = np.random.permutation(len(normal_windows))
    train_normal = normal_windows[perm[:n_train]]
    test_normal = normal_windows[perm[n_train:]]

    # Extract features
    train_features = np.array([extract_features(w) for w in train_normal])
    test_normal_features = np.array([extract_features(w) for w in test_normal])
    fault_features = np.array([extract_features(w) for w in fault_windows])

    # Calibrate encoder on healthy features
    encoder.calibrate(train_features)

    # Train NATIVA
    for feat in train_features:
        spike_matrix = encoder.encode(feat)
        old = sys.stdout; sys.stdout = io.StringIO()
        try: net.feed(spike_matrix, mode="train")
        finally: sys.stdout = old

    # Test
    all_scores, all_labels = [], []

    for feat in test_normal_features:
        spike_matrix = encoder.encode(feat)
        old = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(spike_matrix, mode="test")
        finally: sys.stdout = old
        s = np.array(res['surprise'])
        all_scores.append(float(np.mean(s)) if len(s) > 0 else 0.0)
        all_labels.append(0)

    for feat in fault_features:
        spike_matrix = encoder.encode(feat)
        old = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(spike_matrix, mode="test")
        finally: sys.stdout = old
        s = np.array(res['surprise'])
        all_scores.append(float(np.mean(s)) if len(s) > 0 else 0.0)
        all_labels.append(1)

    try: return roc_auc_score(np.array(all_labels), np.array(all_scores))
    except: return 0.5


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("  NATIVA — FEATURE-SPIKE EXPERIMENT")
    print("  22 FFT features → rate-coded spikes → same SNN")
    print("  Question: Is the gap from encoding or from learning?")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    results = []
    t0 = time.time()

    for load in [0, 1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"  ⚙️  CHARGE : {load} HP ({NORMAL_FILES[load]['rpm']} RPM)")
        print(f"{'='*60}")

        normal_sig = _load_signal(NORMAL_FILES[load]['id'])
        if normal_sig is None:
            continue
        normal_windows = _segment(normal_sig)

        for ftype in ['Ball', 'Inner', 'Outer']:
            for fsize in ['007', '014', '021']:
                fid = FAULT_FILES[ftype][fsize].get(load)
                if fid is None:
                    continue
                fault_sig = _load_signal(fid)
                if fault_sig is None:
                    continue
                fault_windows = _segment(fault_sig)
                cond_name = f"{ftype} {fsize}\" {load}HP"

                auc = run_feature_spike(normal_windows, fault_windows, cond_name)
                print(f"   {cond_name:<25}  AUC={auc:.4f}")
                results.append({
                    'condition': cond_name, 'fault_type': ftype,
                    'fault_size': fsize, 'load_hp': load,
                    'auc_feature_spike': float(auc),
                })

    elapsed = time.time() - t0

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  📋 FEATURE-SPIKE SUMMARY")
    print(f"{'='*70}")

    aucs = [r['auc_feature_spike'] for r in results]
    print(f"\n  Mean AUC (Feature-Spike NATIVA): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Min: {np.min(aucs):.4f}  Max: {np.max(aucs):.4f}")
    print(f"\n  Recall: NATIVA 8-band encoder = 0.951")
    print(f"  Recall: AE/IF/SVM on 22 features = 1.000")
    print(f"  Now:    NATIVA on 22 features     = {np.mean(aucs):.4f}")

    if np.mean(aucs) > 0.98:
        print(f"\n  ✅ CONCLUSION: SNN is NOT the bottleneck. The 8-band encoder was.")
    elif np.mean(aucs) > 0.95:
        print(f"\n  ⚠️  CONCLUSION: Feature-spike improves but doesn't close the gap fully.")
        print(f"      Rate coding may lose information too.")
    else:
        print(f"\n  ❌ CONCLUSION: Feature quality alone doesn't explain the gap.")

    # Per load
    print(f"\n  Per load:")
    for load in [0, 1, 2, 3]:
        ld = [r['auc_feature_spike'] for r in results if r['load_hp'] == load]
        if ld:
            print(f"    {load}HP  AUC={np.mean(ld):.4f} ± {np.std(ld):.4f}")

    # --- Report ---
    _generate_report(results, output_dir)

    # --- JSON ---
    output = {
        'experiment': 'Feature-Spike: 22 FFT features rate-coded into NATIVA',
        'hypothesis': 'If AUC ≈ 1.0, the SNN learning is good and the 8-band encoder was the bottleneck',
        'n_conditions': len(results),
        'summary': {
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'min_auc': float(np.min(aucs)),
            'max_auc': float(np.max(aucs)),
        },
        'comparison': {
            'nativa_8band': 0.951,
            'ae_22features': 1.000,
            'nativa_22features_spike': float(np.mean(aucs)),
        },
        'conditions': results,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = os.path.join(output_dir, 'feature_spike_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 JSON: {json_path}")
    print(f"⏱️  Temps: {elapsed:.0f}s")
    print("\n✅ Feature-Spike experiment terminé.")


def _generate_report(results, output_dir):
    """Comparison bar chart: 8-band vs feature-spike vs baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Feature-Spike Experiment: Is the Gap from Encoding or Learning?",
                 fontsize=13, fontweight='bold')

    # Left: heatmap per condition
    ax = axes[0]
    row_labels = []
    for ftype in ['Ball', 'Inner', 'Outer']:
        for fsize in ['007', '014', '021']:
            row_labels.append(f"{ftype} {fsize}\"")
    col_labels = ['0HP', '1HP', '2HP', '3HP']

    grid = np.full((9, 4), np.nan)
    for r in results:
        for fi, ftype in enumerate(['Ball', 'Inner', 'Outer']):
            for si, fsize in enumerate(['007', '014', '021']):
                if r['fault_type'] == ftype and r['fault_size'] == fsize:
                    li = [0, 1, 2, 3].index(r['load_hp'])
                    grid[fi*3 + si, li] = r['auc_feature_spike']

    im = ax.imshow(grid, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(9))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(f'NATIVA + 22 Feature-Spikes\n(mean AUC = {np.nanmean(grid):.3f})')

    for i in range(9):
        for j in range(4):
            val = grid[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.75 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, label='AUC-ROC', shrink=0.8)

    # Right: bar comparison
    ax2 = axes[1]
    methods = ['NATIVA\n(8-band spikes)', 'NATIVA\n(22-feature spikes)',
               'Autoencoder\n(22 features)', 'Isolation Forest\n(22 features)']
    aucs_all = [r['auc_feature_spike'] for r in results]
    means = [0.951, np.mean(aucs_all), 1.000, 1.000]
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#95a5a6']

    bars = ax2.bar(methods, means, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0.8, 1.02)
    ax2.set_ylabel('Mean AUC-ROC')
    ax2.set_title('Encoding vs Learning:\nWhere is the bottleneck?')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect')
    ax2.axhline(y=0.951, color='red', linestyle='--', alpha=0.3, label='8-band baseline')

    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, 'feature_spike_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport: {path}")


if __name__ == "__main__":
    main()
