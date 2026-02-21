"""
NATIVA — Kuramoto Ablation Study on CWRU
==========================================
Compare NATIVA with and without Kuramoto phase coherence modulation.

This tells us: does Kuramoto actually help anomaly detection, or is it dead weight?

Protocol:
    - Condition A: Full NATIVA (use_kuramoto=True)  — reuses existing results
    - Condition B: NATIVA without Kuramoto (use_kuramoto=False)
    - Same 36 CWRU conditions, same encoding, same seed
    - Compare AUC-ROC per condition, per load, overall

Output:
    - results/ablation_kuramoto.json
    - results/ablation_kuramoto.png (side-by-side heatmaps)
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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

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

# --- Reuse CWRU catalog from benchmark_cwru_multi ---
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
        print(f"   ❌ Missing {file_id}.mat — run benchmark_cwru_multi.py first")
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


def run_one_condition(normal_windows, fault_windows, condition_name, use_kuramoto=True):
    """Run NATIVA on one condition with Kuramoto ON or OFF."""
    params = LIFNeuron(tau_m=50.0, V_th=0.3, R_m=5.0, t_ref=2.0)
    cfg = NativaConfig(
        n_neurons=100, n_output_classes=2, neuron_params=params,
        use_kuramoto=use_kuramoto, kuramoto_coupling=2.0,
        weight_norm_target=100.0, use_adaptive_thresh=True,
        thresh_increment=0.5, seed=42
    )
    net = NativaNetwork(cfg)
    encoder = MultiScaleSpikeEncoder(n_bands=8, n_time_steps=64)

    # Split 50/50
    np.random.seed(42)
    n_train = len(normal_windows) // 2
    perm = np.random.permutation(len(normal_windows))
    train_normal = normal_windows[perm[:n_train]]
    test_normal = normal_windows[perm[n_train:]]

    # Calibrate & train
    encoder.calibrate(train_normal)
    for w in train_normal:
        old = sys.stdout; sys.stdout = io.StringIO()
        try: net.feed(encoder.encode(w), mode="train")
        finally: sys.stdout = old

    # Test
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

    scores, labels = np.array(all_scores), np.array(all_labels)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5

    return float(auc)


def main():
    print("=" * 70)
    print("  NATIVA — KURAMOTO ABLATION STUDY (CWRU, 36 conditions)")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    results = []  # list of {condition, fault_type, fault_size, load_hp, auc_on, auc_off}
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

                # Kuramoto ON
                auc_on = run_one_condition(normal_windows, fault_windows, cond_name, use_kuramoto=True)
                # Kuramoto OFF
                auc_off = run_one_condition(normal_windows, fault_windows, cond_name, use_kuramoto=False)

                delta = auc_on - auc_off
                arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")

                print(f"   {cond_name:<25}  ON={auc_on:.4f}  OFF={auc_off:.4f}  Δ={delta:+.4f} {arrow}")
                results.append({
                    'condition': cond_name, 'fault_type': ftype,
                    'fault_size': fsize, 'load_hp': load,
                    'auc_kuramoto_on': auc_on, 'auc_kuramoto_off': auc_off,
                    'delta': delta,
                })

    elapsed = time.time() - t0

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  📋 ABLATION SUMMARY")
    print(f"{'='*70}")

    aucs_on = [r['auc_kuramoto_on'] for r in results]
    aucs_off = [r['auc_kuramoto_off'] for r in results]
    deltas = [r['delta'] for r in results]

    print(f"\n  Mean AUC (Kuramoto ON):   {np.mean(aucs_on):.4f} ± {np.std(aucs_on):.4f}")
    print(f"  Mean AUC (Kuramoto OFF):  {np.mean(aucs_off):.4f} ± {np.std(aucs_off):.4f}")
    print(f"  Mean Δ (ON - OFF):        {np.mean(deltas):+.4f}")
    print(f"  Conditions where ON > OFF: {sum(1 for d in deltas if d > 0.01)}/{len(deltas)}")
    print(f"  Conditions where OFF > ON: {sum(1 for d in deltas if d < -0.01)}/{len(deltas)}")
    print(f"  Conditions ≈ equal:        {sum(1 for d in deltas if abs(d) <= 0.01)}/{len(deltas)}")

    # Per load
    print(f"\n  Par charge moteur :")
    for load in [0, 1, 2, 3]:
        ld = [r for r in results if r['load_hp'] == load]
        if ld:
            on = np.mean([r['auc_kuramoto_on'] for r in ld])
            off = np.mean([r['auc_kuramoto_off'] for r in ld])
            print(f"    {load}HP  ON={on:.4f}  OFF={off:.4f}  Δ={on-off:+.4f}")

    # --- Generate report ---
    _generate_ablation_report(results, output_dir)

    # --- Save JSON ---
    output = {
        'experiment': 'Kuramoto Ablation on CWRU',
        'n_conditions': len(results),
        'summary': {
            'mean_auc_on': float(np.mean(aucs_on)),
            'mean_auc_off': float(np.mean(aucs_off)),
            'mean_delta': float(np.mean(deltas)),
            'n_on_better': int(sum(1 for d in deltas if d > 0.01)),
            'n_off_better': int(sum(1 for d in deltas if d < -0.01)),
            'n_equal': int(sum(1 for d in deltas if abs(d) <= 0.01)),
        },
        'conditions': results,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = os.path.join(output_dir, 'ablation_kuramoto.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 JSON: {json_path}")
    print(f"⏱️  Temps: {elapsed:.0f}s")
    print("\n✅ Ablation terminée.")


def _generate_ablation_report(results, output_dir):
    """Side-by-side heatmaps: Kuramoto ON vs OFF."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("NATIVA — Kuramoto Ablation (CWRU)\nTop: Kuramoto ON  |  Bottom: Kuramoto OFF",
                 fontsize=14, fontweight='bold')

    for row, (label, key) in enumerate([("Kuramoto ON", 'auc_kuramoto_on'),
                                         ("Kuramoto OFF", 'auc_kuramoto_off')]):
        for col, ftype in enumerate(['Ball', 'Inner', 'Outer']):
            ax = axes[row, col]
            sizes = ['007', '014', '021']
            loads = [0, 1, 2, 3]
            grid = np.full((len(sizes), len(loads)), np.nan)

            for r in results:
                if r['fault_type'] == ftype:
                    si = sizes.index(r['fault_size'])
                    li = loads.index(r['load_hp'])
                    grid[si, li] = r[key]

            im = ax.imshow(grid, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
            ax.set_xticks(range(len(loads)))
            ax.set_xticklabels([f'{l}HP' for l in loads])
            ax.set_yticks(range(len(sizes)))
            ax.set_yticklabels([f'{s}"' for s in sizes])
            ax.set_title(f'{ftype} — {label}')
            if col == 0: ax.set_ylabel('Fault Size')
            if row == 1: ax.set_xlabel('Load')

            for i in range(len(sizes)):
                for j in range(len(loads)):
                    val = grid[i, j]
                    if not np.isnan(val):
                        color = 'white' if val < 0.75 else 'black'
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                fontsize=9, fontweight='bold', color=color)

    fig.colorbar(im, ax=axes, label='AUC-ROC', shrink=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, 'ablation_kuramoto.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport ablation: {path}")


if __name__ == "__main__":
    main()
