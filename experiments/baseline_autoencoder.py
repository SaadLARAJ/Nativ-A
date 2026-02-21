"""
NATIVA — Autoencoder Baseline Comparison (CWRU)
=================================================
Fair comparison: unsupervised Dense Autoencoder vs NATIVA on same 36 conditions.

Both methods:
    - See ONLY healthy data at training time (unsupervised)
    - Are evaluated on healthy + fault windows
    - Use AUC-ROC as metric

Autoencoder approach:
    - Input: 22 hand-crafted features per window (same as RF baseline)
      [RMS, peak, crest, kurtosis, skewness, std, MAE, 10 FFT, 5 band energies]
    - Architecture: Dense 22 → 8 → 22 (bottleneck)
    - Training: Adam-like gradient descent, MSE loss, on healthy windows only
    - Anomaly score: reconstruction MSE per window

Additionally tests:
    - Isolation Forest (sklearn) on same features
    - One-Class SVM (sklearn) on same features

Output:
    - results/baseline_comparison.json
    - results/baseline_comparison.png
"""

import numpy as np
import os
import sys
import json
import io
import time

from scipy.io import loadmat
from scipy.signal import stft
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

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
# CWRU CATALOG (same as other benchmarks)
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
# FEATURE EXTRACTION (same 22 features as RF baseline in the paper)
# =====================================================================

def extract_features(window):
    """Extract 22 hand-crafted features from a vibration window."""
    # Time-domain (7)
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(np.abs(window))
    crest = peak / (rms + 1e-10)
    kurt = float(np.mean((window - np.mean(window))**4) / (np.std(window)**4 + 1e-10))
    skew = float(np.mean((window - np.mean(window))**3) / (np.std(window)**3 + 1e-10))
    std = np.std(window)
    mae = np.mean(np.abs(window))

    # Frequency-domain: FFT magnitudes (10)
    fft_mag = np.abs(np.fft.rfft(window))
    fft_mag = fft_mag / (np.max(fft_mag) + 1e-10)  # normalize
    n_fft = len(fft_mag)
    indices = np.linspace(0, n_fft - 1, 10, dtype=int)
    fft_features = fft_mag[indices]

    # Band energies (5)
    n_bands = 5
    band_size = n_fft // n_bands
    band_energies = np.array([
        np.mean(fft_mag[i*band_size:(i+1)*band_size]**2)
        for i in range(n_bands)
    ])

    return np.concatenate([[rms, peak, crest, kurt, skew, std, mae],
                           fft_features, band_energies])


def extract_features_batch(windows):
    """Extract features for all windows."""
    return np.array([extract_features(w) for w in windows])


# =====================================================================
# DENSE AUTOENCODER (pure NumPy, no PyTorch/TF dependency)
# =====================================================================

class DenseAutoencoder:
    """
    Minimal dense autoencoder: 22 → 8 → 22
    Trained with Adam on MSE loss.
    Anomaly score = reconstruction MSE.
    """
    def __init__(self, input_dim=22, hidden_dim=8, lr=0.001, epochs=100, seed=42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = rng.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, input_dim) * scale
        self.b2 = np.zeros(input_dim)
        self.lr = lr
        self.epochs = epochs
        self.scaler = StandardScaler()

        # Adam state
        self._m = {k: np.zeros_like(v) for k, v in
                   [('W1', self.W1), ('b1', self.b1), ('W2', self.W2), ('b2', self.b2)]}
        self._v = {k: np.zeros_like(v) for k, v in
                   [('W1', self.W1), ('b1', self.b1), ('W2', self.W2), ('b2', self.b2)]}
        self._t = 0

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _forward(self, X):
        self._z1 = X @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        return self._z2  # linear output

    def _adam_update(self, param_name, param, grad, beta1=0.9, beta2=0.999, eps=1e-8):
        self._m[param_name] = beta1 * self._m[param_name] + (1 - beta1) * grad
        self._v[param_name] = beta2 * self._v[param_name] + (1 - beta2) * grad**2
        m_hat = self._m[param_name] / (1 - beta1**self._t)
        v_hat = self._v[param_name] / (1 - beta2**self._t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X_train):
        """Train on healthy data only."""
        X = self.scaler.fit_transform(X_train)
        n = len(X)

        for epoch in range(self.epochs):
            self._t += 1
            # Forward
            X_hat = self._forward(X)
            # MSE loss
            error = X_hat - X  # (n, 22)

            # Backward
            dL_dz2 = error / n  # (n, 22)
            dW2 = self._a1.T @ dL_dz2
            db2 = np.sum(dL_dz2, axis=0)

            dL_da1 = dL_dz2 @ self.W2.T
            dL_dz1 = dL_da1 * self._relu_grad(self._z1)
            dW1 = X.T @ dL_dz1
            db1 = np.sum(dL_dz1, axis=0)

            # Adam updates
            self.W1 = self._adam_update('W1', self.W1, dW1)
            self.b1 = self._adam_update('b1', self.b1, db1)
            self.W2 = self._adam_update('W2', self.W2, dW2)
            self.b2 = self._adam_update('b2', self.b2, db2)

    def score(self, X):
        """Anomaly score = reconstruction MSE per sample."""
        X_scaled = self.scaler.transform(X)
        X_hat = self._forward(X_scaled)
        return np.mean((X_hat - X_scaled)**2, axis=1)


# =====================================================================
# NATIVA (same as other benchmarks)
# =====================================================================

class MultiScaleSpikeEncoder:
    def __init__(self, n_bands=8, n_time_steps=64):
        self.n_bands = n_bands
        self.n_time_steps = n_time_steps
        self.global_max = None

    def calibrate(self, windows):
        all_bands = np.concatenate([self._bands(w) for w in windows], axis=0)
        self.global_max = np.maximum(np.percentile(all_bands, 99, axis=0), 1e-10)

    def encode(self, window, threshold=0.15):
        bands = self._bands(window)
        return (bands / self.global_max[np.newaxis, :] > threshold).astype(np.float64)

    def _bands(self, window):
        nperseg = min(128, len(window) // 4)
        f, t, Zxx = stft(window, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=nperseg//2, return_onesided=True)
        power = np.abs(Zxx) ** 2
        n_freq = power.shape[0]
        bsz = max(1, n_freq // self.n_bands)
        bands = np.zeros((power.shape[1], self.n_bands))
        for b in range(self.n_bands):
            s, e = b * bsz, min((b+1) * bsz, n_freq)
            if s < n_freq:
                bands[:, b] = np.mean(power[s:e, :], axis=0)
        if bands.shape[0] != self.n_time_steps:
            x_old = np.linspace(0, 1, bands.shape[0])
            x_new = np.linspace(0, 1, self.n_time_steps)
            bands = interp1d(x_old, bands, axis=0, kind='linear')(x_new)
        return bands


def run_nativa(normal_windows, fault_windows, train_normal, test_normal):
    """Run NATIVA and return AUC."""
    params = LIFNeuron(tau_m=50.0, V_th=0.3, R_m=5.0, t_ref=2.0)
    cfg = NativaConfig(
        n_neurons=100, n_output_classes=2, neuron_params=params,
        use_kuramoto=True, kuramoto_coupling=2.0,
        weight_norm_target=100.0, use_adaptive_thresh=True,
        thresh_increment=0.5, seed=42
    )
    net = NativaNetwork(cfg)
    encoder = MultiScaleSpikeEncoder(n_bands=8, n_time_steps=64)

    encoder.calibrate(train_normal)
    for w in train_normal:
        old = sys.stdout; sys.stdout = io.StringIO()
        try: net.feed(encoder.encode(w), mode="train")
        finally: sys.stdout = old

    all_scores, all_labels = [], []
    for w in test_normal:
        old = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(encoder.encode(w), mode="test")
        finally: sys.stdout = old
        s = np.array(res['surprise'])
        all_scores.append(float(np.mean(s)) if len(s) > 0 else 0.0)
        all_labels.append(0)
    for w in fault_windows:
        old = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(encoder.encode(w), mode="test")
        finally: sys.stdout = old
        s = np.array(res['surprise'])
        all_scores.append(float(np.mean(s)) if len(s) > 0 else 0.0)
        all_labels.append(1)

    try: return roc_auc_score(np.array(all_labels), np.array(all_scores))
    except: return 0.5


def run_autoencoder(train_features, test_normal_features, fault_features):
    """Run Dense AE and return AUC."""
    ae = DenseAutoencoder(input_dim=train_features.shape[1], hidden_dim=8, lr=0.002, epochs=200)
    ae.fit(train_features)

    normal_scores = ae.score(test_normal_features)
    fault_scores = ae.score(fault_features)

    scores = np.concatenate([normal_scores, fault_scores])
    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(fault_scores))])
    try: return roc_auc_score(labels, scores)
    except: return 0.5


def run_isolation_forest(train_features, test_normal_features, fault_features):
    """Run Isolation Forest and return AUC."""
    iforest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iforest.fit(train_features)

    normal_scores = -iforest.score_samples(test_normal_features)  # higher = more anomalous
    fault_scores = -iforest.score_samples(fault_features)

    scores = np.concatenate([normal_scores, fault_scores])
    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(fault_scores))])
    try: return roc_auc_score(labels, scores)
    except: return 0.5


def run_ocsvm(train_features, test_normal_features, fault_features):
    """Run One-Class SVM and return AUC."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    ocsvm.fit(train_scaled)

    normal_scores = -ocsvm.score_samples(scaler.transform(test_normal_features))
    fault_scores = -ocsvm.score_samples(scaler.transform(fault_features))

    scores = np.concatenate([normal_scores, fault_scores])
    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(fault_scores))])
    try: return roc_auc_score(labels, scores)
    except: return 0.5


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("  NATIVA — BASELINE COMPARISON (CWRU, 36 conditions)")
    print("  Methods: NATIVA | Dense AE | Isolation Forest | OC-SVM")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    results = []
    t0 = time.time()

    for load in [0, 1, 2, 3]:
        print(f"\n{'='*65}")
        print(f"  ⚙️  CHARGE : {load} HP ({NORMAL_FILES[load]['rpm']} RPM)")
        print(f"{'='*65}")

        normal_sig = _load_signal(NORMAL_FILES[load]['id'])
        if normal_sig is None:
            print(f"   ❌ Signal normal manquant pour {load}HP")
            continue
        normal_windows = _segment(normal_sig)

        # Split 50/50 (same seed as NATIVA benchmark)
        np.random.seed(42)
        n_train = len(normal_windows) // 2
        perm = np.random.permutation(len(normal_windows))
        train_normal = normal_windows[perm[:n_train]]
        test_normal = normal_windows[perm[n_train:]]

        # Features for ML baselines
        train_features = extract_features_batch(train_normal)
        test_normal_features = extract_features_batch(test_normal)

        for ftype in ['Ball', 'Inner', 'Outer']:
            for fsize in ['007', '014', '021']:
                fid = FAULT_FILES[ftype][fsize].get(load)
                if fid is None:
                    continue
                fault_sig = _load_signal(fid)
                if fault_sig is None:
                    continue
                fault_windows = _segment(fault_sig)
                fault_features = extract_features_batch(fault_windows)
                cond_name = f"{ftype} {fsize}\" {load}HP"

                # Run all 4 methods
                auc_nativa = run_nativa(normal_windows, fault_windows, train_normal, test_normal)
                auc_ae = run_autoencoder(train_features, test_normal_features, fault_features)
                auc_if = run_isolation_forest(train_features, test_normal_features, fault_features)
                auc_svm = run_ocsvm(train_features, test_normal_features, fault_features)

                # Find winner
                aucs = {'NATIVA': auc_nativa, 'AE': auc_ae, 'IF': auc_if, 'SVM': auc_svm}
                winner = max(aucs, key=aucs.get)

                print(f"   {cond_name:<22}  NAT={auc_nativa:.3f}  AE={auc_ae:.3f}  "
                      f"IF={auc_if:.3f}  SVM={auc_svm:.3f}  ← {winner}")

                results.append({
                    'condition': cond_name, 'fault_type': ftype,
                    'fault_size': fsize, 'load_hp': load,
                    'auc_nativa': float(auc_nativa),
                    'auc_autoencoder': float(auc_ae),
                    'auc_isolation_forest': float(auc_if),
                    'auc_ocsvm': float(auc_svm),
                })

    elapsed = time.time() - t0

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  📋 BASELINE COMPARISON SUMMARY")
    print(f"{'='*70}")

    methods = [('NATIVA', 'auc_nativa'), ('Dense AE', 'auc_autoencoder'),
               ('Isolation Forest', 'auc_isolation_forest'), ('OC-SVM', 'auc_ocsvm')]

    print(f"\n{'Method':<20} {'Mean AUC':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'Wins':>6}")
    print(f"  {'-'*62}")

    summary = {}
    for name, key in methods:
        aucs = [r[key] for r in results]
        wins = sum(1 for r in results if r[key] == max(r['auc_nativa'], r['auc_autoencoder'],
                   r['auc_isolation_forest'], r['auc_ocsvm']))
        summary[name] = {
            'mean': float(np.mean(aucs)), 'std': float(np.std(aucs)),
            'min': float(np.min(aucs)), 'max': float(np.max(aucs)),
            'wins': int(wins)
        }
        print(f"  {name:<20} {np.mean(aucs):>10.4f} {np.std(aucs):>8.4f} "
              f"{np.min(aucs):>8.4f} {np.max(aucs):>8.4f} {wins:>6}")

    # Per load
    print(f"\n  Per load (mean AUC):")
    print(f"  {'Load':<8}", end="")
    for name, _ in methods:
        print(f"  {name:>12}", end="")
    print()
    for load in [0, 1, 2, 3]:
        ld = [r for r in results if r['load_hp'] == load]
        print(f"  {load}HP     ", end="")
        for _, key in methods:
            m = np.mean([r[key] for r in ld])
            print(f"  {m:>12.4f}", end="")
        print()

    # --- Generate report ---
    _generate_comparison_report(results, methods, output_dir)

    # --- Save JSON ---
    output = {
        'experiment': 'Baseline Comparison on CWRU',
        'methods': ['NATIVA (SNN)', 'Dense Autoencoder', 'Isolation Forest', 'One-Class SVM'],
        'n_conditions': len(results),
        'summary': summary,
        'conditions': results,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = os.path.join(output_dir, 'baseline_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 JSON: {json_path}")
    print(f"⏱️  Temps: {elapsed:.0f}s")
    print("\n✅ Comparaison terminée.")


def _generate_comparison_report(results, methods, output_dir):
    """4-panel heatmap: one per method."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CWRU Unsupervised Baseline Comparison — AUC-ROC per Condition",
                 fontsize=14, fontweight='bold')

    method_list = [('NATIVA (SNN)', 'auc_nativa'), ('Dense Autoencoder', 'auc_autoencoder'),
                   ('Isolation Forest', 'auc_isolation_forest'), ('One-Class SVM', 'auc_ocsvm')]

    # Build condition labels: 9 rows (3 types × 3 sizes), 4 cols (loads)
    row_labels = []
    for ftype in ['Ball', 'Inner', 'Outer']:
        for fsize in ['007', '014', '021']:
            row_labels.append(f"{ftype} {fsize}\"")
    col_labels = ['0HP', '1HP', '2HP', '3HP']

    for idx, (name, key) in enumerate(method_list):
        ax = axes[idx // 2, idx % 2]
        grid = np.full((9, 4), np.nan)

        for r in results:
            ri = 0
            for fi, ftype in enumerate(['Ball', 'Inner', 'Outer']):
                for si, fsize in enumerate(['007', '014', '021']):
                    if r['fault_type'] == ftype and r['fault_size'] == fsize:
                        li = [0, 1, 2, 3].index(r['load_hp'])
                        grid[fi*3 + si, li] = r[key]

        im = ax.imshow(grid, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
        ax.set_xticks(range(4))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(9))
        ax.set_yticklabels(row_labels, fontsize=8)
        mean_auc = np.nanmean(grid)
        ax.set_title(f'{name}\n(mean AUC = {mean_auc:.3f})', fontsize=11)

        for i in range(9):
            for j in range(4):
                val = grid[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.75 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, fontweight='bold', color=color)

    fig.colorbar(im, ax=axes, label='AUC-ROC', shrink=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport comparaison: {path}")


if __name__ == "__main__":
    main()
