"""
NATIVA — Mel-Scale Encoder Experiment
=======================================
Replace linear 8-band encoder with Mel-scale (logarithmic) frequency bands.

Hypothesis:
    Linear bands = 750 Hz each on CWRU (0-6kHz / 8 bands).
    Bearing fault frequencies (BPFO ~106Hz, BPFI ~160Hz, BSF ~70Hz) all fall in
    band 0 (0-750Hz), masked by normal operating energy.

    Mel-scale concentrates resolution in low frequencies where faults live:
    Band 0: 0-100 Hz,  Band 1: 100-250 Hz,  Band 2: 250-500 Hz, ...
    This should separate fault harmonics from shaft rotation energy.

Protocol:
    Compare LINEAR vs MEL encoder on:
    1. CWRU 36 conditions (focus on 1HP)
    2. Paderborn 6 bearings

Output:
    - results/mel_encoder_results.json
    - results/mel_encoder_report.png
"""

import numpy as np
import os
import sys
import json
import io
import time
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
# CWRU CATALOG
# =====================================================================
CWRU_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'cwru')
CWRU_SR = 12000

CWRU_NORMAL = {
    0: {'id': 97}, 1: {'id': 98}, 2: {'id': 99}, 3: {'id': 100},
}
CWRU_FAULTS = {
    'Ball':  {'007':{0:118,1:119,2:120,3:121},'014':{0:185,1:186,2:187,3:188},'021':{0:222,1:223,2:224,3:225}},
    'Inner': {'007':{0:105,1:106,2:107,3:108},'014':{0:169,1:170,2:171,3:172},'021':{0:209,1:210,2:211,3:212}},
    'Outer': {'007':{0:130,1:131,2:132,3:133},'014':{0:197,1:198,2:199,3:200},'021':{0:234,1:235,2:236,3:237}},
}

# =====================================================================
# PADERBORN CATALOG
# =====================================================================
PADER_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'paderborn')
PADER_SR = 64000
PADER_HEALTHY = ['K001', 'K002']
PADER_DAMAGED = {
    'Outer (KA)': ['KA01', 'KA03', 'KA05'],
    'Inner (KI)': ['KI01', 'KI03', 'KI05'],
}


# =====================================================================
# DATA LOADERS
# =====================================================================

def _load_cwru(file_id):
    fpath = os.path.join(CWRU_DATA, f"{file_id}.mat")
    if not os.path.exists(fpath): return None
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

def _load_paderborn(mat_path):
    try:
        mat = loadmat(mat_path)
    except: return None
    keys = [k for k in mat if not k.startswith('__')]
    if not keys: return None
    try:
        Y = mat[keys[0]]['Y'][0, 0]
        for ch in range(Y['Name'].shape[1]):
            name = Y['Name'][0, ch]
            if name.size > 0 and 'vibration' in str(name.flat[0]).lower():
                sig = Y['Data'][0, ch].flatten()
                if len(sig) > 10000: return sig
    except: pass
    return None

def _get_paderborn_files(code):
    d = os.path.join(PADER_DATA, code, code)
    if not os.path.exists(d): d = os.path.join(PADER_DATA, code)
    files = glob.glob(os.path.join(d, '*.mat'))
    filtered = [f for f in files if 'N15_M07_F10' in os.path.basename(f)]
    return sorted(filtered if filtered else files)

def _segment(signal, ws=1024, overlap=0.5):
    step = int(ws * (1 - overlap))
    n = (len(signal) - ws) // step
    return np.array([signal[i*step : i*step + ws] for i in range(n)])


# =====================================================================
# ENCODERS
# =====================================================================

def _hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def _mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)


class LinearEncoder:
    """Original 8-band linear encoder."""
    def __init__(self, n_bands=8, n_time=64, sr=12000):
        self.n_bands = n_bands
        self.n_time = n_time
        self.sr = sr
        self.global_max = None

    def calibrate(self, windows):
        all_b = np.concatenate([self._bands(w) for w in windows], axis=0)
        self.global_max = np.maximum(np.percentile(all_b, 99, axis=0), 1e-10)

    def encode(self, window, th=0.15):
        b = self._bands(window)
        return (b / self.global_max[np.newaxis, :] > th).astype(np.float64)

    def _bands(self, window):
        nperseg = min(128 if self.sr <= 12000 else 256, len(window)//4)
        f, t, Zxx = stft(window, fs=self.sr, nperseg=nperseg, noverlap=nperseg//2,
                         return_onesided=True)
        power = np.abs(Zxx)**2
        nf = power.shape[0]
        bsz = max(1, nf // self.n_bands)
        bands = np.zeros((power.shape[1], self.n_bands))
        for b in range(self.n_bands):
            s, e = b*bsz, min((b+1)*bsz, nf)
            if s < nf: bands[:, b] = np.mean(power[s:e, :], axis=0)
        if bands.shape[0] != self.n_time:
            xo = np.linspace(0, 1, bands.shape[0])
            xn = np.linspace(0, 1, self.n_time)
            bands = interp1d(xo, bands, axis=0, kind='linear')(xn)
        return bands


class MelEncoder:
    """Mel-scale (logarithmic) 8-band encoder.
    
    Concentrates frequency resolution in the low-frequency range where
    bearing fault characteristic frequencies live (BPFO, BPFI, BSF).
    
    For CWRU (12kHz, Nyquist=6kHz), the 8 Mel bands are approximately:
        Band 0:    0 -  135 Hz  (shaft rotation, FTF)
        Band 1:  135 -  310 Hz  (BPFO ~106Hz harmonics, BSF ~70Hz harmonics)
        Band 2:  310 -  545 Hz  (BPFI ~160Hz harmonics)
        Band 3:  545 -  870 Hz  
        Band 4:  870 - 1320 Hz
        Band 5: 1320 - 1950 Hz
        Band 6: 1950 - 2830 Hz
        Band 7: 2830 - 6000 Hz  (structural resonances)
    """
    def __init__(self, n_bands=8, n_time=64, sr=12000):
        self.n_bands = n_bands
        self.n_time = n_time
        self.sr = sr
        self.global_max = None
        
        # Compute Mel-spaced band boundaries
        nyquist = sr / 2
        mel_low = _hz_to_mel(0)
        mel_high = _hz_to_mel(nyquist)
        mel_points = np.linspace(mel_low, mel_high, n_bands + 1)
        self.hz_points = _mel_to_hz(mel_points)  # Hz boundaries

    def calibrate(self, windows):
        all_b = np.concatenate([self._bands(w) for w in windows], axis=0)
        self.global_max = np.maximum(np.percentile(all_b, 99, axis=0), 1e-10)

    def encode(self, window, th=0.15):
        b = self._bands(window)
        return (b / self.global_max[np.newaxis, :] > th).astype(np.float64)

    def _bands(self, window):
        nperseg = min(128 if self.sr <= 12000 else 256, len(window)//4)
        f, t, Zxx = stft(window, fs=self.sr, nperseg=nperseg, noverlap=nperseg//2,
                         return_onesided=True)
        power = np.abs(Zxx)**2
        
        bands = np.zeros((power.shape[1], self.n_bands))
        for b in range(self.n_bands):
            lo = self.hz_points[b]
            hi = self.hz_points[b + 1]
            # Find frequency bin indices
            idx = np.where((f >= lo) & (f < hi))[0]
            if len(idx) > 0:
                bands[:, b] = np.mean(power[idx, :], axis=0)
        
        if bands.shape[0] != self.n_time:
            xo = np.linspace(0, 1, bands.shape[0])
            xn = np.linspace(0, 1, self.n_time)
            bands = interp1d(xo, bands, axis=0, kind='linear')(xn)
        return bands


# =====================================================================
# RUN NATIVA
# =====================================================================

def run_nativa(normal_windows, fault_windows, encoder):
    """Run NATIVA with a given encoder, return AUC."""
    params = LIFNeuron(tau_m=50.0, V_th=0.3, R_m=5.0, t_ref=2.0)
    cfg = NativaConfig(
        n_neurons=100, n_output_classes=2, neuron_params=params,
        use_kuramoto=True, kuramoto_coupling=2.0,
        weight_norm_target=100.0, use_adaptive_thresh=True,
        thresh_increment=0.5, seed=42
    )
    net = NativaNetwork(cfg)

    np.random.seed(42)
    n_train = len(normal_windows) // 2
    perm = np.random.permutation(len(normal_windows))
    train = normal_windows[perm[:n_train]]
    test_n = normal_windows[perm[n_train:]]

    encoder.calibrate(train)
    for w in train:
        old = sys.stdout; sys.stdout = io.StringIO()
        try: net.feed(encoder.encode(w), mode="train")
        finally: sys.stdout = old

    scores, labels = [], []
    for w in test_n:
        old = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(encoder.encode(w), mode="test")
        finally: sys.stdout = old
        s = np.array(res['surprise'])
        scores.append(float(np.mean(s)) if len(s) > 0 else 0.0)
        labels.append(0)
    for w in fault_windows:
        old = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(encoder.encode(w), mode="test")
        finally: sys.stdout = old
        s = np.array(res['surprise'])
        scores.append(float(np.mean(s)) if len(s) > 0 else 0.0)
        labels.append(1)

    try: return roc_auc_score(np.array(labels), np.array(scores))
    except: return 0.5


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("  NATIVA — MEL-SCALE ENCODER EXPERIMENT")
    print("  Linear vs Mel-scale frequency bands")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ---- CWRU ----
    print("\n" + "=" * 70)
    print("  PART 1: CWRU (36 conditions)")
    print("=" * 70)

    cwru_results = []
    for load in [0, 1, 2, 3]:
        normal_sig = _load_cwru(CWRU_NORMAL[load]['id'])
        if normal_sig is None: continue
        normal_win = _segment(normal_sig, ws=1024)

        print(f"\n  ⚙️  {load}HP:")
        for ftype in ['Ball', 'Inner', 'Outer']:
            for fsize in ['007', '014', '021']:
                fid = CWRU_FAULTS[ftype][fsize].get(load)
                if fid is None: continue
                fsig = _load_cwru(fid)
                if fsig is None: continue
                fwin = _segment(fsig, ws=1024)
                cond = f"{ftype} {fsize}\" {load}HP"

                lin_enc = LinearEncoder(n_bands=8, n_time=64, sr=CWRU_SR)
                mel_enc = MelEncoder(n_bands=8, n_time=64, sr=CWRU_SR)

                auc_lin = run_nativa(normal_win, fwin, lin_enc)
                auc_mel = run_nativa(normal_win, fwin, mel_enc)
                delta = auc_mel - auc_lin
                arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")

                print(f"    {cond:<22}  LIN={auc_lin:.3f}  MEL={auc_mel:.3f}  Δ={delta:+.3f} {arrow}")
                cwru_results.append({
                    'dataset': 'CWRU', 'condition': cond,
                    'fault_type': ftype, 'fault_size': fsize, 'load_hp': load,
                    'auc_linear': float(auc_lin), 'auc_mel': float(auc_mel),
                    'delta': float(delta),
                })

    # ---- PADERBORN ----
    print("\n" + "=" * 70)
    print("  PART 2: PADERBORN (6 bearings)")
    print("=" * 70)

    pader_results = []
    # Load healthy
    healthy_sigs = []
    for code in PADER_HEALTHY:
        for mf in _get_paderborn_files(code)[:5]:
            s = _load_paderborn(mf)
            if s is not None: healthy_sigs.append(s)
    if healthy_sigs:
        healthy_all = np.concatenate(healthy_sigs)
        pader_normal = _segment(healthy_all, ws=4096)
        print(f"\n  Healthy windows: {len(pader_normal)}")

        for dtype, codes in PADER_DAMAGED.items():
            print(f"\n  🔧 {dtype}:")
            for code in codes:
                fault_sigs = []
                for mf in _get_paderborn_files(code)[:5]:
                    s = _load_paderborn(mf)
                    if s is not None: fault_sigs.append(s)
                if not fault_sigs: continue
                fault_all = np.concatenate(fault_sigs)
                fwin = _segment(fault_all, ws=4096)

                lin_enc = LinearEncoder(n_bands=8, n_time=64, sr=PADER_SR)
                mel_enc = MelEncoder(n_bands=8, n_time=64, sr=PADER_SR)

                auc_lin = run_nativa(pader_normal, fwin, lin_enc)
                auc_mel = run_nativa(pader_normal, fwin, mel_enc)
                delta = auc_mel - auc_lin
                arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")

                print(f"    {code:<10}  LIN={auc_lin:.3f}  MEL={auc_mel:.3f}  Δ={delta:+.3f} {arrow}")
                pader_results.append({
                    'dataset': 'Paderborn', 'bearing': code, 'damage_type': dtype,
                    'auc_linear': float(auc_lin), 'auc_mel': float(auc_mel),
                    'delta': float(delta),
                })

    elapsed = time.time() - t0
    all_results = cwru_results + pader_results

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print("  📋 MEL-SCALE ENCODER SUMMARY")
    print(f"{'='*70}")

    # CWRU
    if cwru_results:
        cl = [r['auc_linear'] for r in cwru_results]
        cm = [r['auc_mel'] for r in cwru_results]
        print(f"\n  CWRU (36 conditions):")
        print(f"    Linear: {np.mean(cl):.4f} ± {np.std(cl):.4f}")
        print(f"    Mel:    {np.mean(cm):.4f} ± {np.std(cm):.4f}")
        print(f"    Δ mean: {np.mean(cm)-np.mean(cl):+.4f}")

        # Per load
        for load in [0, 1, 2, 3]:
            ld_l = [r['auc_linear'] for r in cwru_results if r['load_hp'] == load]
            ld_m = [r['auc_mel'] for r in cwru_results if r['load_hp'] == load]
            if ld_l:
                print(f"    {load}HP: LIN={np.mean(ld_l):.4f} → MEL={np.mean(ld_m):.4f} (Δ={np.mean(ld_m)-np.mean(ld_l):+.4f})")

    # Paderborn
    if pader_results:
        pl = [r['auc_linear'] for r in pader_results]
        pm = [r['auc_mel'] for r in pader_results]
        print(f"\n  Paderborn (6 bearings):")
        print(f"    Linear: {np.mean(pl):.4f}")
        print(f"    Mel:    {np.mean(pm):.4f}")
        print(f"    Δ mean: {np.mean(pm)-np.mean(pl):+.4f}")

    # Report
    _generate_report(cwru_results, pader_results, output_dir)

    # JSON
    output = {
        'experiment': 'Mel-Scale Encoder Comparison',
        'hypothesis': 'Mel-scale bands concentrate resolution around fault frequencies, improving detection',
        'cwru': {
            'n_conditions': len(cwru_results),
            'mean_linear': float(np.mean(cl)) if cwru_results else 0,
            'mean_mel': float(np.mean(cm)) if cwru_results else 0,
        },
        'paderborn': {
            'n_bearings': len(pader_results),
            'mean_linear': float(np.mean(pl)) if pader_results else 0,
            'mean_mel': float(np.mean(pm)) if pader_results else 0,
        },
        'cwru_conditions': cwru_results,
        'paderborn_bearings': pader_results,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = os.path.join(output_dir, 'mel_encoder_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 JSON: {json_path}")
    print(f"⏱️  Temps: {elapsed:.0f}s")
    print("\n✅ Mel encoder experiment terminé.")


def _generate_report(cwru, pader, output_dir):
    """Visual comparison: Linear vs Mel on both datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("NATIVA — Linear vs Mel-Scale Encoder", fontsize=14, fontweight='bold')

    # Left: CWRU heatmap (Mel)
    ax = axes[0]
    row_labels = []
    for ft in ['Ball', 'Inner', 'Outer']:
        for fs in ['007', '014', '021']:
            row_labels.append(f"{ft} {fs}\"")
    grid_mel = np.full((9, 4), np.nan)
    grid_lin = np.full((9, 4), np.nan)
    for r in cwru:
        for fi, ft in enumerate(['Ball', 'Inner', 'Outer']):
            for si, fs in enumerate(['007', '014', '021']):
                if r['fault_type'] == ft and r['fault_size'] == fs:
                    li = r['load_hp']
                    grid_mel[fi*3+si, li] = r['auc_mel']
                    grid_lin[fi*3+si, li] = r['auc_linear']

    im = ax.imshow(grid_mel, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_title(f'CWRU — Mel Encoder\n(mean={np.nanmean(grid_mel):.3f})')
    ax.set_xticks(range(4)); ax.set_xticklabels(['0HP','1HP','2HP','3HP'])
    ax.set_yticks(range(9)); ax.set_yticklabels(row_labels, fontsize=7)
    for i in range(9):
        for j in range(4):
            v = grid_mel[i,j]
            if not np.isnan(v):
                c = 'white' if v < 0.75 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6, fontweight='bold', color=c)

    # Middle: Delta heatmap (Mel - Linear)
    ax2 = axes[1]
    delta = grid_mel - grid_lin
    im2 = ax2.imshow(delta, cmap='RdBu', vmin=-0.2, vmax=0.2, aspect='auto')
    ax2.set_title('CWRU — Δ (Mel - Linear)')
    ax2.set_xticks(range(4)); ax2.set_xticklabels(['0HP','1HP','2HP','3HP'])
    ax2.set_yticks(range(9)); ax2.set_yticklabels(row_labels, fontsize=7)
    for i in range(9):
        for j in range(4):
            v = delta[i,j]
            if not np.isnan(v):
                c = 'white' if abs(v) > 0.1 else 'black'
                ax2.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=6, fontweight='bold', color=c)
    fig.colorbar(im2, ax=ax2, label='Δ AUC', shrink=0.6)

    # Right: Paderborn bar chart
    ax3 = axes[2]
    if pader:
        names = [r['bearing'] for r in pader]
        lin_aucs = [r['auc_linear'] for r in pader]
        mel_aucs = [r['auc_mel'] for r in pader]
        x = np.arange(len(names))
        w = 0.35
        ax3.bar(x - w/2, lin_aucs, w, label='Linear', color='#e74c3c', alpha=0.8)
        ax3.bar(x + w/2, mel_aucs, w, label='Mel', color='#2ecc71', alpha=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(names, rotation=45)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel('AUC-ROC')
        ax3.set_title(f'Paderborn — Linear vs Mel\n(Lin={np.mean(lin_aucs):.3f}, Mel={np.mean(mel_aucs):.3f})')
        ax3.legend()
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    fig.colorbar(im, ax=ax, label='AUC-ROC', shrink=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, 'mel_encoder_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport: {path}")


if __name__ == "__main__":
    main()
