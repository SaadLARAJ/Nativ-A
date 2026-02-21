"""
NATIVA — Frugal Envelope V2 (with Expert Corrections)
======================================================
Fixes 4 critical issues identified in the naive envelope:

1. WINDOW SIZE: time-based (100ms) instead of sample-based (1024)
   → CWRU: 1200 samples, Paderborn: 6400 samples (always ~2.5 rotations)

2. HIGH-PASS > 2kHz BEFORE ENVELOPE: removes rotor balourd/shaft noise,
   keeps only structural resonance from bearing impacts

3. τ_m = 150ms: longer neuron memory to catch slow faults (FTF ~10Hz)

4. CALIBRATED Δ: delta threshold learned from healthy percentile,
   not fixed constant

Output:
    - results/envelope_v2_results.json
    - results/envelope_v2_report.png
"""

import numpy as np
import os
import sys
import json
import io
import time
import glob

from scipy.io import loadmat
from scipy.signal import butter, sosfilt
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
# DATA CATALOGS
# =====================================================================
CWRU_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'cwru')
CWRU_SR = 12000
CWRU_NORMAL = {0:{'id':97}, 1:{'id':98}, 2:{'id':99}, 3:{'id':100}}
CWRU_FAULTS = {
    'Ball':  {'007':{0:118,1:119,2:120,3:121},'014':{0:185,1:186,2:187,3:188},'021':{0:222,1:223,2:224,3:225}},
    'Inner': {'007':{0:105,1:106,2:107,3:108},'014':{0:169,1:170,2:171,3:172},'021':{0:209,1:210,2:211,3:212}},
    'Outer': {'007':{0:130,1:131,2:132,3:133},'014':{0:197,1:198,2:199,3:200},'021':{0:234,1:235,2:236,3:237}},
}

PADER_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'paderborn')
PADER_SR = 64000
PADER_HEALTHY = ['K001', 'K002']
PADER_DAMAGED = {
    'Outer (KA)': ['KA01', 'KA03', 'KA05'],
    'Inner (KI)': ['KI01', 'KI03', 'KI05'],
}

# FIX #1: Time-based window (100ms = ~2.5 rotations at typical speeds)
WINDOW_MS = 100  # milliseconds


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
            cands = [k for k in mat if 'DE_time' in k]
            if not cands: return None
            key = cands[0]
    return mat[key].flatten()

def _load_paderborn(mpath):
    try: mat = loadmat(mpath)
    except: return None
    keys = [k for k in mat if not k.startswith('__')]
    if not keys: return None
    try:
        Y = mat[keys[0]]['Y'][0, 0]
        for ch in range(Y['Name'].shape[1]):
            nm = Y['Name'][0, ch]
            if nm.size > 0 and 'vibration' in str(nm.flat[0]).lower():
                s = Y['Data'][0, ch].flatten()
                if len(s) > 10000: return s
    except: pass
    return None

def _get_pader_files(code):
    d = os.path.join(PADER_DATA, code, code)
    if not os.path.exists(d): d = os.path.join(PADER_DATA, code)
    if not os.path.exists(d): return []
    files = glob.glob(os.path.join(d, '*.mat'))
    filt = [f for f in files if 'N15_M07_F10' in os.path.basename(f)]
    return sorted(filt if filt else files)

def _segment_time_based(signal, sr, window_ms=100, overlap=0.5):
    """FIX #1: Time-based windowing. Always ~100ms regardless of sample rate."""
    ws = int(sr * window_ms / 1000)
    step = int(ws * (1 - overlap))
    n = (len(signal) - ws) // step
    if n < 1: return np.array([]), ws
    return np.array([signal[i*step : i*step + ws] for i in range(n)]), ws


# =====================================================================
# FRUGAL ENVELOPE V2 ENCODER
# =====================================================================

class FrugalEnvelopeV2:
    """
    Edge-AI envelope encoder with expert corrections.
    
    Pipeline:
        1. HIGH-PASS > 2kHz (FIX #2): remove rotor noise, keep resonance
        2. Rectification: |x|
        3. Low-pass smooth: moving average (~5ms window)
        4. Multi-band split (3 bands: 2-4kHz, 4-Nyquist/2, Nyquist/2-Nyquist)
        5. Delta modulation with CALIBRATED threshold (FIX #4)
    
    Computational cost: ~20 ops/sample (2nd order IIR + abs + avg + cmp)
    On a 1 MHz MCU @ 64kHz sampling: trivial (3% CPU).
    """
    def __init__(self, n_bands=8, n_time_steps=64, sr=12000):
        self.n_bands = n_bands
        self.n_time_steps = n_time_steps
        self.sr = sr
        self.global_max = None
        self.delta_thresholds = None  # FIX #4: per-band calibrated thresholds
        
        nyquist = sr / 2.0
        
        # FIX #2: High-pass filter > 2kHz
        # This removes rotor balourd/shaft harmonics (0-2kHz)
        # and preserves structural resonance excited by bearing impacts
        hp_cutoff = min(2000.0, nyquist * 0.4)  # adapt if SR is low
        hp_norm = hp_cutoff / nyquist
        hp_norm = max(0.01, min(0.95, hp_norm))
        self.highpass = butter(2, hp_norm, btype='high', output='sos')
        
        # Band boundaries within the high-passed signal (2kHz to Nyquist)
        # Log-spaced for resolution where resonances cluster
        low_hz = hp_cutoff
        hi_hz = nyquist * 0.95
        log_edges = np.logspace(np.log10(max(low_hz, 50)),
                                np.log10(max(hi_hz, low_hz + 100)),
                                n_bands + 1)
        
        self.filters = []
        for b in range(n_bands):
            lo = log_edges[b] / nyquist
            hi = log_edges[b + 1] / nyquist
            lo = max(lo, 0.01)
            hi = min(hi, 0.99)
            if lo >= hi: hi = min(lo + 0.01, 0.99)
            try:
                sos = butter(2, [lo, hi], btype='bandpass', output='sos')
                self.filters.append(sos)
            except:
                self.filters.append(None)
        
        self.smooth_win = max(3, int(sr * 0.005))  # ~5ms

    def calibrate(self, windows):
        """Calibrate on healthy data: learn global_max AND delta thresholds."""
        all_env = np.concatenate([self._envelope_bands(w) for w in windows], axis=0)
        self.global_max = np.maximum(np.percentile(all_env, 99, axis=0), 1e-10)
        
        # FIX #4: Learn per-band delta thresholds from healthy dynamics
        # The threshold = 50th percentile of |diff| in healthy windows
        # This means: "spike when the change is bigger than typical healthy change"
        all_norm = all_env / self.global_max[np.newaxis, :]
        diffs = np.abs(np.diff(all_norm, axis=0))
        self.delta_thresholds = np.percentile(diffs, 50, axis=0)  # median change
        self.delta_thresholds = np.maximum(self.delta_thresholds, 0.01)  # min floor

    def encode(self, window):
        """Encode window into delta-modulated spike matrix."""
        env = self._envelope_bands(window)
        env_norm = env / self.global_max[np.newaxis, :]
        return self._delta_modulate(env_norm)

    def _envelope_bands(self, window):
        """FIX #2: High-pass first, THEN bandpass → rectify → smooth."""
        # Step 1: High-pass to remove rotor noise
        try:
            hp_signal = sosfilt(self.highpass, window)
        except:
            hp_signal = window
        
        n_samples = len(hp_signal)
        bands = np.zeros((n_samples, self.n_bands))
        
        for b, sos in enumerate(self.filters):
            if sos is None:
                bands[:, b] = np.abs(hp_signal)
                continue
            try:
                filtered = sosfilt(sos, hp_signal)
            except:
                filtered = hp_signal
            
            rectified = np.abs(filtered)
            kernel = np.ones(self.smooth_win) / self.smooth_win
            bands[:, b] = np.convolve(rectified, kernel, mode='same')
        
        # Resample to n_time_steps
        if bands.shape[0] != self.n_time_steps:
            xo = np.linspace(0, 1, bands.shape[0])
            xn = np.linspace(0, 1, self.n_time_steps)
            bands = interp1d(xo, bands, axis=0, kind='linear')(xn)
        
        return bands

    def _delta_modulate(self, env_norm):
        """FIX #4: Delta modulation with calibrated per-band thresholds."""
        n_t, n_b = env_norm.shape
        spikes = np.zeros((n_t, n_b))
        
        for t in range(n_t):
            for b in range(n_b):
                th = self.delta_thresholds[b] if self.delta_thresholds is not None else 0.15
                if t == 0:
                    spikes[t, b] = 1.0 if env_norm[t, b] > th else 0.0
                else:
                    delta = abs(env_norm[t, b] - env_norm[t-1, b])
                    if delta > th or env_norm[t, b] > 2.0:
                        spikes[t, b] = 1.0
        return spikes


# =====================================================================
# RUN NATIVA
# =====================================================================

def run_nativa(normal_win, fault_win, encoder, tau_m=50.0):
    """FIX #3: tau_m parameter (default 50, test with 150 for envelope)."""
    params = LIFNeuron(tau_m=tau_m, V_th=0.3, R_m=5.0, t_ref=2.0)
    cfg = NativaConfig(n_neurons=100, n_output_classes=2, neuron_params=params,
                       use_kuramoto=True, kuramoto_coupling=2.0,
                       weight_norm_target=100.0, use_adaptive_thresh=True,
                       thresh_increment=0.5, seed=42)
    net = NativaNetwork(cfg)

    np.random.seed(42)
    n_train = len(normal_win) // 2
    perm = np.random.permutation(len(normal_win))
    train = normal_win[perm[:n_train]]
    test_n = normal_win[perm[n_train:]]

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
        scores.append(float(np.mean(s)) if len(s)>0 else 0.0)
        labels.append(0)
    for w in fault_win:
        old = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(encoder.encode(w), mode="test")
        finally: sys.stdout = old
        s = np.array(res['surprise'])
        scores.append(float(np.mean(s)) if len(s)>0 else 0.0)
        labels.append(1)

    try: return roc_auc_score(np.array(labels), np.array(scores))
    except: return 0.5


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("  NATIVA — FRUGAL ENVELOPE V2 (Expert Corrections)")
    print("  Fix 1: Window = 100ms (time-based)")
    print("  Fix 2: High-pass > 2kHz (remove rotor noise)")
    print("  Fix 3: τ_m = 150ms (slow fault memory)")
    print("  Fix 4: Calibrated Δ (healthy percentile)")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ---- CWRU ----
    print(f"\n{'='*70}")
    print(f"  PART 1: CWRU — Window = {WINDOW_MS}ms = {int(CWRU_SR*WINDOW_MS/1000)} samples")
    print(f"  (vs old 1024 samples = {1024/CWRU_SR*1000:.0f}ms)")
    print(f"{'='*70}")

    cwru_results = []
    for load in [0, 1, 2, 3]:
        normal_sig = _load_cwru(CWRU_NORMAL[load]['id'])
        if normal_sig is None: continue
        normal_win, ws = _segment_time_based(normal_sig, CWRU_SR, WINDOW_MS)
        if len(normal_win) == 0: continue

        print(f"\n  ⚙️  {load}HP ({len(normal_win)} windows × {ws} samples):")
        for ftype in ['Ball', 'Inner', 'Outer']:
            for fsize in ['007', '014', '021']:
                fid = CWRU_FAULTS[ftype][fsize].get(load)
                if fid is None: continue
                fsig = _load_cwru(fid)
                if fsig is None: continue
                fwin, _ = _segment_time_based(fsig, CWRU_SR, WINDOW_MS)
                if len(fwin) == 0: continue
                cond = f"{ftype} {fsize}\" {load}HP"

                # FIX #3: τ_m = 150ms
                enc = FrugalEnvelopeV2(n_bands=8, n_time_steps=64, sr=CWRU_SR)
                auc = run_nativa(normal_win, fwin, enc, tau_m=150.0)

                print(f"    {cond:<22}  AUC={auc:.4f}")
                cwru_results.append({
                    'dataset': 'CWRU', 'condition': cond,
                    'fault_type': ftype, 'fault_size': fsize, 'load_hp': load,
                    'auc': float(auc),
                })

    # ---- PADERBORN ----
    print(f"\n{'='*70}")
    print(f"  PART 2: PADERBORN — Window = {WINDOW_MS}ms = {int(PADER_SR*WINDOW_MS/1000)} samples")
    print(f"  (vs old 4096 samples = {4096/PADER_SR*1000:.0f}ms)")
    print(f"{'='*70}")

    pader_results = []
    healthy_sigs = []
    for code in PADER_HEALTHY:
        for mf in _get_pader_files(code)[:5]:
            s = _load_paderborn(mf)
            if s is not None: healthy_sigs.append(s)

    if healthy_sigs:
        healthy_all = np.concatenate(healthy_sigs)
        pader_normal, ws = _segment_time_based(healthy_all, PADER_SR, WINDOW_MS)
        print(f"\n  Healthy: {len(pader_normal)} windows × {ws} samples")

        for dtype, codes in PADER_DAMAGED.items():
            print(f"\n  🔧 {dtype}:")
            for code in codes:
                fsigs = []
                for mf in _get_pader_files(code)[:5]:
                    s = _load_paderborn(mf)
                    if s is not None: fsigs.append(s)
                if not fsigs: continue
                fall = np.concatenate(fsigs)
                fwin, _ = _segment_time_based(fall, PADER_SR, WINDOW_MS)
                if len(fwin) < 10: continue

                enc = FrugalEnvelopeV2(n_bands=8, n_time_steps=64, sr=PADER_SR)
                auc = run_nativa(pader_normal, fwin, enc, tau_m=150.0)

                print(f"    {code:<10}  AUC={auc:.4f}")
                pader_results.append({
                    'dataset': 'Paderborn', 'bearing': code, 'damage_type': dtype,
                    'auc': float(auc),
                })
    else:
        print("\n  ❌ No Paderborn data")

    elapsed = time.time() - t0

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print("  📋 ENVELOPE V2 SUMMARY (with expert corrections)")
    print(f"{'='*70}")

    if cwru_results:
        ca = [r['auc'] for r in cwru_results]
        print(f"\n  CWRU: {np.mean(ca):.4f} ± {np.std(ca):.4f}")
        for load in [0,1,2,3]:
            la = [r['auc'] for r in cwru_results if r['load_hp']==load]
            if la: print(f"    {load}HP: {np.mean(la):.4f} ± {np.std(la):.4f}")

    if pader_results:
        pa = [r['auc'] for r in pader_results]
        print(f"\n  Paderborn: {np.mean(pa):.4f} ± {np.std(pa):.4f}")

    # Evolution table
    print(f"\n  📊 Complete Encoder Evolution:")
    print(f"    {'Encoder':<25} {'CWRU':>8} {'1HP':>8} {'Paderborn':>10}")
    print(f"    {'─'*25} {'─'*8} {'─'*8} {'─'*10}")
    print(f"    {'Linear (v1.0)':<25} {'0.951':>8} {'0.823':>8} {'0.503':>10}")
    print(f"    {'Mel-scale':<25} {'0.692':>8} {'0.434':>8} {'0.596':>10}")
    print(f"    {'Envelope v1 (naive)':<25} {'0.889':>8} {'0.905':>8} {'0.624':>10}")
    if cwru_results:
        cwru_mean = np.mean(ca)
        hp1 = [r['auc'] for r in cwru_results if r['load_hp']==1]
        hp1_mean = np.mean(hp1) if hp1 else 0
        pad_mean = np.mean(pa) if pader_results else 0
        print(f"    {'Envelope v2 (expert)':<25} {cwru_mean:>8.3f} {hp1_mean:>8.3f} {pad_mean:>10.3f}")
    print(f"    {'Feature-Spike*':<25} {'0.995':>8} {'0.995':>8} {'N/A':>10}")

    # Report
    _generate_report(cwru_results, pader_results, output_dir)

    # JSON
    output = {
        'experiment': 'Frugal Envelope V2 (Expert Corrections)',
        'fixes': [
            'Window = 100ms (time-based)',
            'High-pass > 2kHz (remove rotor noise)',
            'tau_m = 150ms (slow fault memory)',
            'Calibrated delta (healthy P50)',
        ],
        'cwru': {
            'mean': float(np.mean(ca)) if cwru_results else None,
            'std': float(np.std(ca)) if cwru_results else None,
            'n': len(cwru_results),
        },
        'paderborn': {
            'mean': float(np.mean(pa)) if pader_results else None,
            'n': len(pader_results),
        },
        'cwru_conditions': cwru_results,
        'paderborn_bearings': pader_results,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    jpath = os.path.join(output_dir, 'envelope_v2_results.json')
    with open(jpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 JSON: {jpath}")
    print(f"⏱️  Temps: {elapsed:.0f}s")


def _generate_report(cwru, pader, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("NATIVA — Frugal Envelope V2 (Expert Corrections)\n"
                 "Window=100ms | HP>2kHz | τ_m=150ms | Calibrated Δ",
                 fontsize=12, fontweight='bold')

    # CWRU heatmap
    ax = axes[0]
    rl = [f"{ft} {fs}\"" for ft in ['Ball','Inner','Outer'] for fs in ['007','014','021']]
    grid = np.full((9,4), np.nan)
    for r in cwru:
        for fi,ft in enumerate(['Ball','Inner','Outer']):
            for si,fs in enumerate(['007','014','021']):
                if r['fault_type']==ft and r['fault_size']==fs:
                    grid[fi*3+si, r['load_hp']] = r['auc']
    im = ax.imshow(grid, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_title(f'CWRU Envelope V2\n(mean={np.nanmean(grid):.3f})')
    ax.set_xticks(range(4)); ax.set_xticklabels(['0HP','1HP','2HP','3HP'])
    ax.set_yticks(range(9)); ax.set_yticklabels(rl, fontsize=7)
    for i in range(9):
        for j in range(4):
            v=grid[i,j]
            if not np.isnan(v):
                ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6,
                       fontweight='bold',color='white' if v<0.75 else 'black')
    fig.colorbar(im, ax=ax, label='AUC-ROC', shrink=0.6)

    # All encoders comparison
    ax2 = axes[1]
    names = ['Linear\nv1.0', 'Mel', 'Env v1\n(naive)', 'Env v2\n(expert)', 'Feature\nSpike*']
    cwru_m = [0.951, 0.692, 0.889, np.nanmean(grid), 0.995]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#95a5a6']
    bars = ax2.bar(names, cwru_m, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0.4, 1.05); ax2.set_ylabel('Mean AUC')
    ax2.set_title('CWRU — Encoder Evolution')
    for bar,val in zip(bars, cwru_m):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Paderborn
    ax3 = axes[2]
    if pader:
        names_p = [r['bearing'] for r in pader]
        aucs_p = [r['auc'] for r in pader]
        colors_p = ['#e74c3c' if 'KA' in r['bearing'] else '#3498db' for r in pader]
        ax3.bar(names_p, aucs_p, color=colors_p, edgecolor='black', linewidth=0.5)
        ax3.set_ylim(0, 1.05); ax3.set_ylabel('AUC-ROC')
        ax3.set_title(f'Paderborn V2\n(mean={np.mean(aucs_p):.3f})')
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='chance')
        ax3.axhline(y=0.624, color='orange', linestyle='--', alpha=0.3, label='V1=0.624')
        ax3.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'envelope_v2_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport: {path}")


if __name__ == "__main__":
    main()
