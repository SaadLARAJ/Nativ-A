"""
NATIVA — Frugal Envelope Encoder Experiment
=============================================
The NATIVA 2.0 direction: physically-principled, hardware-compatible encoding.

Background:
    - Linear encoder: good for CWRU (broadband resonance), fails on Paderborn
    - Mel encoder: helps Paderborn slightly, destroys CWRU Ball faults
    - Full Hilbert/FFT envelope: would work but defeats SWaP argument
    
Frugal Envelope approach:
    1. Rectification: |signal|  (1 operation per sample, zero cost)
    2. Multi-band bandpass: split into bands THEN rectify each band
    3. Low-pass filter: moving average (simple IIR, edge-compatible)
    4. Delta modulation: spike when envelope rises/falls above threshold
    
    The SNN's LIF neurons become the spectral analyzer:
    - They receive delta-modulated envelope from multiple bands
    - Their time constants (τ_m) naturally integrate over different durations
    - STDP learns which band envelopes correlate with "normal" patterns
    - Free Energy detects when envelope rhythm changes (= fault)

    This is fundamentally different from "feature-spike" (where the FFT
    did all the work). Here, the SNN does the temporal pattern recognition.

Output:
    - results/envelope_results.json
    - results/envelope_report.png
"""

import numpy as np
import os
import sys
import json
import io
import time
import glob

from scipy.io import loadmat
from scipy.signal import stft, butter, filtfilt, sosfilt
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
# DATA CATALOGS (same as mel_encoder.py)
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

def _segment(signal, ws=1024, overlap=0.5):
    step = int(ws * (1 - overlap))
    n = (len(signal) - ws) // step
    if n < 1: return np.array([])
    return np.array([signal[i*step : i*step + ws] for i in range(n)])


# =====================================================================
# FRUGAL ENVELOPE ENCODER
# =====================================================================

class FrugalEnvelopeEncoder:
    """
    Edge-AI compatible envelope encoder.
    
    Pipeline (per band):
        1. Bandpass filter (2nd order Butterworth — IIR, very cheap)
        2. Rectification: |x|  (absolute value, 1 op per sample)
        3. Lowpass smoothing: moving average (edge-compatible)
        4. Delta modulation: spike on significant change
    
    The SNN receives delta-modulated envelopes from N bands.
    Its job: learn the RHYTHM patterns in normal data, detect rhythm changes.
    
    Computational cost per sample:
        - Bandpass: ~10 multiplies (2nd order IIR × N bands)
        - Rectify: N comparisons
        - Smooth: N additions (running average)
        - Delta: N comparisons
        Total: ~15N operations per sample. At N=8: ~120 ops/sample.
        On a 1 MHz microcontroller: trivial.
    """
    def __init__(self, n_bands=8, n_time_steps=64, sr=12000):
        self.n_bands = n_bands
        self.n_time_steps = n_time_steps
        self.sr = sr
        self.global_max = None
        
        # Design bandpass filters (logarithmically spaced)
        nyquist = sr / 2.0
        # Use log-spaced bands optimized for bearing frequencies
        # Low bound: ~20 Hz (below shaft speed), High: Nyquist
        low_hz = 20.0
        hi_hz = min(nyquist * 0.95, nyquist - 10)
        
        # Log-spaced center frequencies
        log_edges = np.logspace(np.log10(low_hz), np.log10(hi_hz), n_bands + 1)
        
        self.filters = []
        for b in range(n_bands):
            lo = log_edges[b] / nyquist
            hi = log_edges[b + 1] / nyquist
            # Clamp to valid range
            lo = max(lo, 0.005)
            hi = min(hi, 0.995)
            if lo >= hi:
                hi = min(lo + 0.01, 0.995)
            try:
                sos = butter(2, [lo, hi], btype='bandpass', output='sos')
                self.filters.append(sos)
            except Exception:
                # Fallback: simple highpass
                self.filters.append(None)
        
        # Smoothing window (in samples) — ~5ms window
        self.smooth_win = max(3, int(sr * 0.005))
    
    def calibrate(self, windows):
        """Calibrate normalization on healthy data."""
        all_envelopes = np.concatenate([self._envelope_bands(w) for w in windows], axis=0)
        self.global_max = np.maximum(np.percentile(all_envelopes, 99, axis=0), 1e-10)
    
    def encode(self, window):
        """Encode window into delta-modulated spike matrix."""
        env = self._envelope_bands(window)
        
        # Normalize by healthy calibration
        env_norm = env / self.global_max[np.newaxis, :]
        
        # Delta modulation: spike when envelope changes significantly
        spikes = self._delta_modulate(env_norm)
        
        return spikes
    
    def _envelope_bands(self, window):
        """Extract envelope from each frequency band.
        
        This is the frugal pipeline:
            bandpass → |x| → moving average
        """
        n_samples = len(window)
        bands = np.zeros((n_samples, self.n_bands))
        
        for b, sos in enumerate(self.filters):
            if sos is None:
                bands[:, b] = np.abs(window)
                continue
            
            # 1. Bandpass filter
            try:
                filtered = sosfilt(sos, window)
            except Exception:
                filtered = window
            
            # 2. Rectify
            rectified = np.abs(filtered)
            
            # 3. Smooth (moving average — edge-compatible)
            kernel = np.ones(self.smooth_win) / self.smooth_win
            smoothed = np.convolve(rectified, kernel, mode='same')
            
            bands[:, b] = smoothed
        
        # Resample to n_time_steps
        if bands.shape[0] != self.n_time_steps:
            xo = np.linspace(0, 1, bands.shape[0])
            xn = np.linspace(0, 1, self.n_time_steps)
            bands = interp1d(xo, bands, axis=0, kind='linear')(xn)
        
        return bands
    
    def _delta_modulate(self, env_norm, threshold=0.15):
        """Delta modulation: spike on significant envelope change.
        
        For each band, at each timestep:
        - Spike = 1 if |envelope[t] - envelope[t-1]| > threshold
          OR envelope[t] > 1.0 (absolute anomaly)
        - Spike = 0 otherwise
        
        This preserves TEMPORAL DYNAMICS, not just amplitude.
        The SNN can then learn the rhythm of these changes.
        """
        n_t, n_b = env_norm.shape
        spikes = np.zeros((n_t, n_b))
        
        for t in range(n_t):
            for b in range(n_b):
                if t == 0:
                    # First timestep: spike if above baseline
                    spikes[t, b] = 1.0 if env_norm[t, b] > threshold else 0.0
                else:
                    delta = abs(env_norm[t, b] - env_norm[t-1, b])
                    # Spike on change OR on high absolute level
                    if delta > threshold or env_norm[t, b] > 1.5:
                        spikes[t, b] = 1.0
        
        return spikes


# =====================================================================
# ORIGINAL LINEAR ENCODER (for comparison)
# =====================================================================

class LinearEncoder:
    def __init__(self, n_bands=8, n_time=64, sr=12000):
        self.n_bands = n_bands; self.n_time = n_time; self.sr = sr
        self.global_max = None
    def calibrate(self, w):
        a = np.concatenate([self._b(x) for x in w], axis=0)
        self.global_max = np.maximum(np.percentile(a, 99, axis=0), 1e-10)
    def encode(self, w, th=0.15):
        b = self._b(w); return (b / self.global_max[np.newaxis,:] > th).astype(np.float64)
    def _b(self, w):
        nperseg = min(128 if self.sr<=12000 else 256, len(w)//4)
        f,t,Z = stft(w, fs=self.sr, nperseg=nperseg, noverlap=nperseg//2, return_onesided=True)
        p = np.abs(Z)**2; nf = p.shape[0]; bsz = max(1,nf//self.n_bands)
        bands = np.zeros((p.shape[1], self.n_bands))
        for b in range(self.n_bands):
            s,e = b*bsz, min((b+1)*bsz, nf)
            if s<nf: bands[:,b] = np.mean(p[s:e,:], axis=0)
        if bands.shape[0]!=self.n_time:
            xo=np.linspace(0,1,bands.shape[0]); xn=np.linspace(0,1,self.n_time)
            bands=interp1d(xo,bands,axis=0,kind='linear')(xn)
        return bands


# =====================================================================
# RUN NATIVA
# =====================================================================

def run_nativa(normal_win, fault_win, encoder):
    params = LIFNeuron(tau_m=50.0, V_th=0.3, R_m=5.0, t_ref=2.0)
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
    print("  NATIVA — FRUGAL ENVELOPE ENCODER")
    print("  |signal| → bandpass → rectify → smooth → delta spikes")
    print("  The SNN becomes the spectral analyzer")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ---- CWRU ----
    print(f"\n{'='*70}")
    print("  PART 1: CWRU (36 conditions)")
    print(f"{'='*70}")

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
                env_enc = FrugalEnvelopeEncoder(n_bands=8, n_time_steps=64, sr=CWRU_SR)

                auc_lin = run_nativa(normal_win, fwin, lin_enc)
                auc_env = run_nativa(normal_win, fwin, env_enc)
                delta = auc_env - auc_lin
                arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")

                print(f"    {cond:<22}  LIN={auc_lin:.3f}  ENV={auc_env:.3f}  Δ={delta:+.3f} {arrow}")
                cwru_results.append({
                    'dataset': 'CWRU', 'condition': cond,
                    'fault_type': ftype, 'fault_size': fsize, 'load_hp': load,
                    'auc_linear': float(auc_lin), 'auc_envelope': float(auc_env),
                    'delta': float(delta),
                })

    # ---- PADERBORN ----
    print(f"\n{'='*70}")
    print("  PART 2: PADERBORN (6 bearings)")
    print(f"{'='*70}")

    pader_results = []
    healthy_sigs = []
    for code in PADER_HEALTHY:
        for mf in _get_pader_files(code)[:5]:
            s = _load_paderborn(mf)
            if s is not None: healthy_sigs.append(s)

    if healthy_sigs:
        healthy_all = np.concatenate(healthy_sigs)
        pader_normal = _segment(healthy_all, ws=4096)
        print(f"\n  Healthy windows: {len(pader_normal)}")

        for dtype, codes in PADER_DAMAGED.items():
            print(f"\n  🔧 {dtype}:")
            for code in codes:
                fsigs = []
                for mf in _get_pader_files(code)[:5]:
                    s = _load_paderborn(mf)
                    if s is not None: fsigs.append(s)
                if not fsigs: continue
                fall = np.concatenate(fsigs)
                fwin = _segment(fall, ws=4096)
                if len(fwin) < 10: continue

                lin_enc = LinearEncoder(n_bands=8, n_time=64, sr=PADER_SR)
                env_enc = FrugalEnvelopeEncoder(n_bands=8, n_time_steps=64, sr=PADER_SR)

                auc_lin = run_nativa(pader_normal, fwin, lin_enc)
                auc_env = run_nativa(pader_normal, fwin, env_enc)
                delta = auc_env - auc_lin
                arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")

                print(f"    {code:<10}  LIN={auc_lin:.3f}  ENV={auc_env:.3f}  Δ={delta:+.3f} {arrow}")
                pader_results.append({
                    'dataset': 'Paderborn', 'bearing': code, 'damage_type': dtype,
                    'auc_linear': float(auc_lin), 'auc_envelope': float(auc_env),
                    'delta': float(delta),
                })
    else:
        print("\n  ❌ No Paderborn data found (download first)")

    elapsed = time.time() - t0

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print("  📋 FRUGAL ENVELOPE SUMMARY")
    print(f"{'='*70}")

    if cwru_results:
        cl = [r['auc_linear'] for r in cwru_results]
        ce = [r['auc_envelope'] for r in cwru_results]
        print(f"\n  CWRU (36 conditions):")
        print(f"    Linear:   {np.mean(cl):.4f} ± {np.std(cl):.4f}")
        print(f"    Envelope: {np.mean(ce):.4f} ± {np.std(ce):.4f}")
        print(f"    Δ mean:   {np.mean(ce)-np.mean(cl):+.4f}")
        for load in [0,1,2,3]:
            ll = [r['auc_linear'] for r in cwru_results if r['load_hp']==load]
            le = [r['auc_envelope'] for r in cwru_results if r['load_hp']==load]
            if ll: print(f"    {load}HP: LIN={np.mean(ll):.4f} → ENV={np.mean(le):.4f} (Δ={np.mean(le)-np.mean(ll):+.4f})")

    if pader_results:
        pl = [r['auc_linear'] for r in pader_results]
        pe = [r['auc_envelope'] for r in pader_results]
        print(f"\n  Paderborn (6 bearings):")
        print(f"    Linear:   {np.mean(pl):.4f}")
        print(f"    Envelope: {np.mean(pe):.4f}")
        print(f"    Δ mean:   {np.mean(pe)-np.mean(pl):+.4f}")

    # Overall comparison
    print(f"\n  📊 Encoder Comparison (all experiments):")
    print(f"    {'Encoder':<20} {'CWRU':>10} {'Paderborn':>12}")
    print(f"    {'Linear (v1.0)':<20} {'0.951':>10} {'0.503':>12}")
    print(f"    {'Mel-scale':<20} {'0.692':>10} {'0.596':>12}")
    if cwru_results:
        cwru_env = np.mean([r['auc_envelope'] for r in cwru_results])
        pader_env = np.mean([r['auc_envelope'] for r in pader_results]) if pader_results else 'N/A'
        print(f"    {'Frugal Envelope':<20} {cwru_env:>10.3f} {pader_env if isinstance(pader_env,str) else f'{pader_env:>12.3f}'}")
        print(f"    {'Feature-Spike*':<20} {'0.995':>10} {'N/A':>12}")
        print(f"    * Feature-spike uses 22 FFT features (not edge-compatible)")

    _generate_report(cwru_results, pader_results, output_dir)

    # JSON
    output = {
        'experiment': 'Frugal Envelope Encoder',
        'method': '|signal| → bandpass(log) → rectify → moving_avg → delta_spikes',
        'edge_cost': '~120 ops/sample (vs ~1000 for FFT)',
        'cwru': {
            'n_conditions': len(cwru_results),
            'mean_linear': float(np.mean(cl)) if cwru_results else None,
            'mean_envelope': float(np.mean(ce)) if cwru_results else None,
        },
        'paderborn': {
            'n_bearings': len(pader_results),
            'mean_linear': float(np.mean(pl)) if pader_results else None,
            'mean_envelope': float(np.mean(pe)) if pader_results else None,
        },
        'cwru_conditions': cwru_results,
        'paderborn_bearings': pader_results,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    jpath = os.path.join(output_dir, 'envelope_results.json')
    with open(jpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 JSON: {jpath}")
    print(f"⏱️  Temps: {elapsed:.0f}s")
    print("\n✅ Frugal Envelope experiment terminé.")


def _generate_report(cwru, pader, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("NATIVA — Frugal Envelope Encoder\n|signal| → bandpass → rectify → smooth → Δ spikes",
                 fontsize=13, fontweight='bold')

    # Left: CWRU envelope heatmap
    ax = axes[0]
    rl = []; 
    for ft in ['Ball','Inner','Outer']:
        for fs in ['007','014','021']: rl.append(f"{ft} {fs}\"")
    grid = np.full((9,4), np.nan)
    for r in cwru:
        for fi,ft in enumerate(['Ball','Inner','Outer']):
            for si,fs in enumerate(['007','014','021']):
                if r['fault_type']==ft and r['fault_size']==fs:
                    grid[fi*3+si, r['load_hp']] = r['auc_envelope']
    im = ax.imshow(grid, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_title(f'CWRU — Frugal Envelope\n(mean={np.nanmean(grid):.3f})')
    ax.set_xticks(range(4)); ax.set_xticklabels(['0HP','1HP','2HP','3HP'])
    ax.set_yticks(range(9)); ax.set_yticklabels(rl, fontsize=7)
    for i in range(9):
        for j in range(4):
            v=grid[i,j]
            if not np.isnan(v):
                ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6,
                       fontweight='bold',color='white' if v<0.75 else 'black')
    fig.colorbar(im, ax=ax, label='AUC-ROC', shrink=0.6)

    # Middle: comparison bar chart (all encoders on CWRU)
    ax2 = axes[1]
    enc_names = ['Linear\n(v1.0)', 'Mel', 'Envelope\n(frugal)', 'Feature\nSpike*']
    cwru_means = [0.951, 0.692, np.nanmean(grid), 0.995]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
    bars = ax2.bar(enc_names, cwru_means, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0.4, 1.05); ax2.set_ylabel('Mean AUC-ROC')
    ax2.set_title('CWRU — All Encoders')
    for bar, val in zip(bars, cwru_means):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.2)

    # Right: Paderborn bar chart
    ax3 = axes[2]
    if pader:
        names = [r['bearing'] for r in pader]
        lin_a = [r['auc_linear'] for r in pader]
        env_a = [r['auc_envelope'] for r in pader]
        x = np.arange(len(names)); w = 0.35
        ax3.bar(x-w/2, lin_a, w, label='Linear', color='#3498db', alpha=0.8)
        ax3.bar(x+w/2, env_a, w, label='Envelope', color='#2ecc71', alpha=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(names, rotation=45)
        ax3.set_ylim(0, 1.05); ax3.set_ylabel('AUC-ROC')
        ax3.set_title(f'Paderborn — Linear vs Envelope\n(Lin={np.mean(lin_a):.3f}, Env={np.mean(env_a):.3f})')
        ax3.legend(); ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'envelope_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport: {path}")


if __name__ == "__main__":
    main()
