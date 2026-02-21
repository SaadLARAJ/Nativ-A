"""
NATIVA — Paderborn Bearing Dataset Benchmark
===============================================
Second dataset validation to prove NATIVA generalizes beyond CWRU.

Dataset: Paderborn University KAt Bearing DataCenter
    - 64 kHz sampling rate
    - 6 healthy bearings (K001–K006), 26 damaged
    - Damage types: KA (outer race), KI (inner race)
    - Each bearing: 20 measurements × 4 seconds
    - Operating condition: N15_M07_F10 (1500 RPM, 0.7 Nm, 1000 N radial)

Protocol (same as CWRU):
    1. Pool all healthy bearings → calibration + test
    2. Test on damaged bearings (unseen)
    3. Report AUC-ROC per damage type

Download: Zenodo (auto)

Output:
    - results/paderborn_results.json
    - results/paderborn_report.png
"""

import numpy as np
import os
import sys
import json
import io
import ssl
import time
import urllib.request
import subprocess
import glob

from scipy.io import loadmat
from scipy.signal import stft
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score

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
# PADERBORN CATALOG
# =====================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'paderborn')
SAMPLE_RATE = 64000  # 64 kHz

ZENODO_BASE = "https://zenodo.org/api/records/15845309/files"

# Healthy bearings (use 2 for speed — ~320MB total instead of 1GB)
HEALTHY_BEARINGS = ['K001', 'K002']

# Damaged bearings — artificially damaged (clean, reproducible)
# Keep compact: 3 outer + 3 inner = 6 bearings (~960MB)
DAMAGED_BEARINGS = {
    'Outer Race (KA)': ['KA01', 'KA03', 'KA05'],
    'Inner Race (KI)': ['KI01', 'KI03', 'KI05'],
}

# Operating condition for file naming
OPERATING_COND = "N15_M07_F10"


def _get_mat_files(bearing_code):
    """Get .mat files for a bearing (already downloaded via curl)."""
    bearing_dir = os.path.join(DATA_DIR, bearing_code, bearing_code)
    if not os.path.exists(bearing_dir):
        # Try flat structure
        bearing_dir = os.path.join(DATA_DIR, bearing_code)
    mat_files = glob.glob(os.path.join(bearing_dir, '*.mat'))
    # Filter only the standard operating condition (N15_M07_F10)
    filtered = [f for f in mat_files if 'N15_M07_F10' in os.path.basename(f)]
    if not filtered:
        filtered = mat_files  # fallback: use all
    return sorted(filtered)


def _load_vibration(mat_path):
    """Load vibration_1 signal from a Paderborn .mat file.
    
    Structure: key → Y.Data[0, 6] = vibration_1 (256001 samples at 64kHz)
    """
    try:
        mat = loadmat(mat_path)
    except Exception:
        return None
    
    data_keys = [k for k in mat if not k.startswith('__')]
    if not data_keys:
        return None
    
    try:
        data = mat[data_keys[0]]
        Y = data['Y'][0, 0]
        n_channels = Y['Name'].shape[1]
        
        # Find vibration_1 channel
        for ch in range(n_channels):
            name = Y['Name'][0, ch]
            if name.size > 0 and 'vibration' in str(name.flat[0]).lower():
                signal = Y['Data'][0, ch].flatten()
                if len(signal) > 10000:
                    return signal
        
        # Fallback: use channel 6 (vibration_1 in standard Paderborn format)
        if n_channels > 6:
            signal = Y['Data'][0, 6].flatten()
            if len(signal) > 10000:
                return signal
    except Exception:
        pass
    
    return None


def _segment(signal, window_size=4096, overlap=0.5):
    """Segment signal into windows. 4096 at 64kHz = 64ms (comparable to CWRU's 85ms)."""
    step = int(window_size * (1 - overlap))
    n = (len(signal) - window_size) // step
    return np.array([signal[i*step : i*step + window_size] for i in range(n)])


# =====================================================================
# ENCODER (same as CWRU but adapted for 64kHz)
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
        nperseg = min(256, len(window) // 4)  # bigger nperseg for 64kHz
        f, t, Zxx = stft(window, fs=SAMPLE_RATE, nperseg=nperseg,
                         noverlap=nperseg//2, return_onesided=True)
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


# =====================================================================
# BENCHMARK
# =====================================================================

def run_paderborn_condition(normal_windows, fault_windows, condition_name):
    """Run NATIVA on one Paderborn condition, return AUC."""
    params = LIFNeuron(tau_m=50.0, V_th=0.3, R_m=5.0, t_ref=2.0)
    cfg = NativaConfig(
        n_neurons=100, n_output_classes=2, neuron_params=params,
        use_kuramoto=True, kuramoto_coupling=2.0,
        weight_norm_target=100.0, use_adaptive_thresh=True,
        thresh_increment=0.5, seed=42
    )
    net = NativaNetwork(cfg)
    encoder = MultiScaleSpikeEncoder(n_bands=8, n_time_steps=64)

    # Split normal 50/50
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

    try:
        return roc_auc_score(np.array(all_labels), np.array(all_scores))
    except ValueError:
        return 0.5


def main():
    print("=" * 70)
    print("  NATIVA — PADERBORN BEARING DATASET BENCHMARK")
    print("  Second dataset validation (after CWRU)")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    results = []
    t0 = time.time()

    # --- Download healthy bearings ---
    print("\n📥 Downloading healthy bearings...")
    healthy_signals = []
    for code in HEALTHY_BEARINGS:
        mat_files = _get_mat_files(code)
        if not mat_files:
            print(f"   ❌ {code}: no files")
            continue
        count = 0
        for mf in sorted(mat_files)[:5]:  # 5 measurements per bearing (enough)
            sig = _load_vibration(mf)
            if sig is not None and len(sig) > 10000:
                healthy_signals.append(sig)
                count += 1
        print(f"   ✅ {code}: {count} signals loaded ({len(mat_files)} total .mat files)")

    if not healthy_signals:
        print("❌ No healthy signals loaded. Aborting.")
        return

    # Concatenate and segment healthy data
    healthy_all = np.concatenate(healthy_signals)
    normal_windows = _segment(healthy_all, window_size=4096, overlap=0.5)
    print(f"\n   Total healthy windows: {len(normal_windows)}")

    # --- Download and test damaged bearings ---
    print("\n📥 Downloading and testing damaged bearings...")

    for damage_type, bearing_codes in DAMAGED_BEARINGS.items():
        print(f"\n{'='*60}")
        print(f"  🔧 {damage_type}")
        print(f"{'='*60}")

        for code in bearing_codes:
            mat_files = _get_mat_files(code)
            if not mat_files:
                print(f"   ❌ {code}: download/extract failed")
                continue

            # Load fault signals
            fault_signals = []
            for mf in sorted(mat_files)[:5]:
                sig = _load_vibration(mf)
                if sig is not None and len(sig) > 10000:
                    fault_signals.append(sig)

            if not fault_signals:
                print(f"   ❌ {code}: no vibration signals found in .mat files")
                continue

            fault_all = np.concatenate(fault_signals)
            fault_windows = _segment(fault_all, window_size=4096, overlap=0.5)

            if len(fault_windows) < 10:
                print(f"   ❌ {code}: too few windows ({len(fault_windows)})")
                continue

            auc = run_paderborn_condition(normal_windows, fault_windows, code)
            print(f"   {code:<10} AUC={auc:.4f}  ({len(fault_windows)} fault windows)")

            results.append({
                'bearing': code,
                'damage_type': damage_type,
                'auc': float(auc),
                'n_fault_windows': len(fault_windows),
            })

    elapsed = time.time() - t0

    if not results:
        print("\n❌ No results. Check download/extraction.")
        return

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  📋 PADERBORN SUMMARY")
    print(f"{'='*70}")

    aucs = [r['auc'] for r in results]
    print(f"\n  Total bearings tested: {len(results)}")
    print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Min: {np.min(aucs):.4f}  Max: {np.max(aucs):.4f}")

    for dt in DAMAGED_BEARINGS.keys():
        dt_aucs = [r['auc'] for r in results if r['damage_type'] == dt]
        if dt_aucs:
            print(f"  {dt}: {np.mean(dt_aucs):.4f} ± {np.std(dt_aucs):.4f}")

    # Cross-dataset summary
    print(f"\n  📊 Cross-Dataset Comparison:")
    print(f"    CWRU (36 conditions):  AUC = 0.951")
    print(f"    Paderborn ({len(results)} bearings): AUC = {np.mean(aucs):.3f}")

    # --- Report ---
    _generate_report(results, output_dir)

    # --- JSON ---
    output = {
        'experiment': 'Paderborn Bearing Dataset Benchmark',
        'dataset': 'Paderborn University KAt DataCenter',
        'sampling_rate': SAMPLE_RATE,
        'window_size': 4096,
        'n_healthy_bearings': len(HEALTHY_BEARINGS),
        'n_healthy_windows': int(len(normal_windows)),
        'n_tested_bearings': len(results),
        'summary': {
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'min_auc': float(np.min(aucs)),
            'max_auc': float(np.max(aucs)),
        },
        'bearings': results,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = os.path.join(output_dir, 'paderborn_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 JSON: {json_path}")
    print(f"⏱️  Temps: {elapsed:.0f}s")
    print("\n✅ Paderborn benchmark terminé.")


def _generate_report(results, output_dir):
    """Bar chart of AUC per bearing."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("NATIVA — Paderborn University Bearing Dataset\nAUC-ROC per Bearing",
                 fontsize=13, fontweight='bold')

    names = [r['bearing'] for r in results]
    aucs = [r['auc'] for r in results]
    colors = ['#e74c3c' if r['damage_type'].startswith('Outer') else '#3498db'
              for r in results]

    bars = ax.bar(names, aucs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUC-ROC')
    ax.axhline(y=np.mean(aucs), color='green', linestyle='--', alpha=0.5,
               label=f'Mean = {np.mean(aucs):.3f}')
    ax.axhline(y=0.951, color='orange', linestyle='--', alpha=0.5,
               label='CWRU mean = 0.951')
    ax.legend()

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Legend: red = outer, blue = inner
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Outer Race (KA)'),
                       Patch(facecolor='#3498db', label='Inner Race (KI)')]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='lower right')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(output_dir, 'paderborn_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport: {path}")


if __name__ == "__main__":
    main()
