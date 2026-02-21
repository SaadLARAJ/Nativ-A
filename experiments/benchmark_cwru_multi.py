"""
NATIVA-Learn — CWRU Multi-Condition Benchmark
===============================================
Test de robustesse : NATIVA sur TOUTES les conditions CWRU.

Conditions testées :
    - 4 charges moteur : 0HP, 1HP, 2HP, 3HP
    - 3 tailles de défaut : 0.007", 0.014", 0.021"
    - 3 types de défaut : Ball, Inner Race, Outer Race
    - 1 baseline normale par charge

Protocole :
    Pour chaque charge moteur :
        1. Calibrer l'encodeur + NATIVA sur les fenêtres NORMALES de cette charge
        2. Tester sur les fenêtres DÉFAUT de cette charge (tous types + tailles)
        3. Reporter AUC-ROC, F1, Recall

Référence fichiers CWRU :
    https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
    File IDs validés contre la documentation officielle CWRU et multiples papers.

Usage:
    python benchmark_cwru_multi.py
"""

import numpy as np
import os
import sys
import json
import ssl
import io
import urllib.request
import time

from scipy.io import loadmat
from scipy.signal import stft
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
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
# 1. CWRU FILE CATALOG
# =====================================================================
# Source: CWRU Bearing Data Center — 12k Drive End
# Validated against official documentation + dozens of published papers
# Format: file_id.mat → key X{file_id}_DE_time
#
# DECISION LOG:
# - On utilise UNIQUEMENT le 12kHz Drive End (le plus standard)
# - Outer Race fault : position @6 (centré, le plus utilisé dans la littérature)
# - Les file IDs sont les numéros officiels du CWRU Data Center

BASE_URL = 'https://engineering.case.edu/sites/default/files'
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cwru')
SAMPLE_RATE = 12000

# Normal baselines par charge moteur
NORMAL_FILES = {
    0: {'id': 97,  'rpm': 1797},  # 0HP
    1: {'id': 98,  'rpm': 1772},  # 1HP
    2: {'id': 99,  'rpm': 1750},  # 2HP
    3: {'id': 100, 'rpm': 1730},  # 3HP
}

# Fault files : fault_type → fault_size → load → file_id
# Ref: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
FAULT_FILES = {
    'Ball': {
        '007': {0: 118, 1: 119, 2: 120, 3: 121},
        '014': {0: 185, 1: 186, 2: 187, 3: 188},
        '021': {0: 222, 1: 223, 2: 224, 3: 225},
    },
    'Inner': {
        '007': {0: 105, 1: 106, 2: 107, 3: 108},
        '014': {0: 169, 1: 170, 2: 171, 3: 172},
        '021': {0: 209, 1: 210, 2: 211, 3: 212},
    },
    'Outer': {  # Position @6 (centered)
        '007': {0: 130, 1: 131, 2: 132, 3: 133},
        '014': {0: 197, 1: 198, 2: 199, 3: 200},
        '021': {0: 234, 1: 235, 2: 236, 3: 237},
    },
}


def _download_file(file_id):
    """Télécharge un fichier CWRU par son ID."""
    os.makedirs(DATA_DIR, exist_ok=True)
    fname = f"{file_id}.mat"
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        return fpath

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"{BASE_URL}/{fname}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as resp, open(fpath, 'wb') as out:
            out.write(resp.read())
        return fpath
    except Exception as e:
        print(f"   ❌ Échec téléchargement {fname}: {e}")
        return None


def _load_signal(file_id):
    """Charge le signal Drive End depuis un fichier CWRU .mat."""
    fpath = _download_file(file_id)
    if fpath is None:
        return None

    try:
        mat = loadmat(fpath)
    except Exception as e:
        print(f"   ❌ Fichier corrompu {file_id}.mat: {e}")
        # Supprimer et re-télécharger
        os.remove(fpath)
        fpath = _download_file(file_id)
        if fpath is None:
            return None
        try:
            mat = loadmat(fpath)
        except Exception as e2:
            print(f"   ❌ Échec définitif {file_id}.mat: {e2}")
            return None

    # Clé standard : X{file_id}_DE_time
    key = f"X{file_id:03d}_DE_time"
    if key not in mat:
        # Variante sans padding
        key2 = f"X{file_id}_DE_time"
        if key2 in mat:
            key = key2
        else:
            # Fallback : chercher toute clé contenant DE_time
            candidates = [k for k in mat.keys() if 'DE_time' in k]
            if not candidates:
                print(f"   ❌ Clé DE_time introuvable dans {file_id}.mat")
                print(f"      Clés dispo: {[k for k in mat if not k.startswith('__')]}")
                return None
            key = candidates[0]

    return mat[key].flatten()


def _segment(signal, window_size=1024, overlap=0.5):
    """Segmente un signal en fenêtres."""
    step = int(window_size * (1 - overlap))
    n = (len(signal) - window_size) // step
    return np.array([signal[i*step : i*step + window_size] for i in range(n)])


# =====================================================================
# 2. MULTI-SCALE SPIKE ENCODER (identique au v2)
# =====================================================================

class MultiScaleSpikeEncoder:
    """
    Encodeur multi-échelle : STFT → 8 bandes → norm globale → spikes.
    Voir benchmark_cwru.py pour la doc détaillée.
    """
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
        f, t, Zxx = stft(window, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=nperseg//2,
                         return_onesided=True)
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
# 3. SINGLE-CONDITION BENCHMARK
# =====================================================================

def benchmark_one_condition(normal_windows, fault_windows, condition_name, verbose=True):
    """
    Benchmark NATIVA sur une condition spécifique.

    Protocol :
        1. Calibre encodeur + réseau sur normal_windows
        2. Teste sur (normal_windows_test + fault_windows)
        3. Retourne métriques

    DECISION :
        - On split 50/50 les fenêtres normales (50% calibration, 50% test)
        - Toutes les fenêtres de défaut vont en test
        - Cela simule le scénario industriel réel
    """
    # Config NATIVA
    params = LIFNeuron(tau_m=50.0, V_th=0.3, R_m=5.0, t_ref=2.0)
    cfg = NativaConfig(
        n_neurons=100, n_output_classes=2, neuron_params=params,
        use_kuramoto=True, kuramoto_coupling=2.0,
        weight_norm_target=100.0, use_adaptive_thresh=True,
        thresh_increment=0.5, seed=42
    )
    net = NativaNetwork(cfg)
    encoder = MultiScaleSpikeEncoder(n_bands=8, n_time_steps=64)

    # Split normales : 50% train, 50% test
    np.random.seed(42)
    n_train = len(normal_windows) // 2
    perm = np.random.permutation(len(normal_windows))
    train_normal = normal_windows[perm[:n_train]]
    test_normal = normal_windows[perm[n_train:]]

    # Calibration
    encoder.calibrate(train_normal)
    for w in train_normal:
        old_out = sys.stdout; sys.stdout = io.StringIO()
        try: net.feed(encoder.encode(w), mode="train")
        finally: sys.stdout = old_out

    # Test
    all_scores, all_labels = [], []

    # Test normales
    for w in test_normal:
        old_out = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(encoder.encode(w), mode="test")
        finally: sys.stdout = old_out
        s = np.array(res['surprise'])
        all_scores.append(float(np.mean(s)) if len(s) > 0 else 0.0)
        all_labels.append(0)

    # Test défauts
    for w in fault_windows:
        old_out = sys.stdout; sys.stdout = io.StringIO()
        try: res = net.feed(encoder.encode(w), mode="test")
        finally: sys.stdout = old_out
        s = np.array(res['surprise'])
        all_scores.append(float(np.mean(s)) if len(s) > 0 else 0.0)
        all_labels.append(1)

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # Métriques
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5

    # Meilleur F1
    best_f1, best_t = 0, 0
    for p in np.arange(5, 95, 2):
        t = np.percentile(scores, p)
        y_pred = (scores >= t).astype(int)
        if np.sum(y_pred) == 0 or np.sum(y_pred) == len(y_pred):
            continue
        f = f1_score(labels, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t

    y_pred = (scores >= best_t).astype(int)
    cm = confusion_matrix(labels, y_pred) if len(np.unique(y_pred)) > 1 else np.array([[0,0],[0,0]])

    result = {
        'condition': condition_name,
        'auc_roc': float(auc),
        'f1': float(best_f1),
        'precision': float(precision_score(labels, y_pred, zero_division=0)),
        'recall': float(recall_score(labels, y_pred, zero_division=0)),
        'n_normal_test': len(test_normal),
        'n_fault_test': len(fault_windows),
        'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
    }

    if verbose:
        print(f"   {condition_name:<30} AUC={auc:.4f}  F1={best_f1:.4f}  "
              f"R={result['recall']:.4f}  FN={result['false_negatives']}  "
              f"FP={result['false_positives']}")

    return result


# =====================================================================
# 4. MAIN — RUN ALL CONDITIONS
# =====================================================================

def main():
    print("=" * 70)
    print("  NATIVA-Learn — CWRU MULTI-CONDITION BENCHMARK")
    print("  3 fault types × 3 sizes × 4 loads = 36 conditions")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    all_results = []
    summary_table = []

    # --- Téléchargement de tous les fichiers ---
    print("\n📥 Téléchargement de tous les fichiers CWRU...")
    n_files = 4 + 36  # 4 normal + 36 fault
    downloaded = 0
    for load, info in NORMAL_FILES.items():
        if _download_file(info['id']):
            downloaded += 1
    for ftype in FAULT_FILES:
        for fsize in FAULT_FILES[ftype]:
            for load, fid in FAULT_FILES[ftype][fsize].items():
                if _download_file(fid):
                    downloaded += 1
    print(f"   ✅ {downloaded}/{n_files} fichiers prêts\n")

    # --- Boucle sur les charges ---
    for load in [0, 1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"  ⚙️  CHARGE MOTEUR : {load} HP ({NORMAL_FILES[load]['rpm']} RPM)")
        print(f"{'='*70}")

        # Charger les fenêtres normales
        normal_sig = _load_signal(NORMAL_FILES[load]['id'])
        if normal_sig is None:
            print(f"   ❌ Signal normal introuvable pour {load}HP — skip")
            continue
        normal_windows = _segment(normal_sig)
        print(f"   Normal: {len(normal_windows)} fenêtres")

        # Tester chaque type × taille de défaut
        for ftype in ['Ball', 'Inner', 'Outer']:
            for fsize in ['007', '014', '021']:
                fid = FAULT_FILES[ftype][fsize].get(load)
                if fid is None:
                    continue

                fault_sig = _load_signal(fid)
                if fault_sig is None:
                    continue
                fault_windows = _segment(fault_sig)

                condition = f"{ftype} {fsize}\" {load}HP"
                result = benchmark_one_condition(
                    normal_windows, fault_windows, condition, verbose=True
                )
                result['fault_type'] = ftype
                result['fault_size'] = fsize
                result['load_hp'] = load
                result['file_id'] = fid
                all_results.append(result)

    # =====================================================================
    # SYNTHÈSE
    # =====================================================================
    print("\n" + "=" * 70)
    print("  📋 SYNTHÈSE MULTI-CONDITION")
    print("=" * 70)

    # Tableau récapitulatif
    print(f"\n  {'Condition':<30} {'AUC':>8} {'F1':>8} {'Recall':>8} {'FN':>5} {'FP':>5}")
    print(f"  {'-'*66}")

    aucs = []
    for r in all_results:
        print(f"  {r['condition']:<30} {r['auc_roc']:>8.4f} {r['f1']:>8.4f} "
              f"{r['recall']:>8.4f} {r['false_negatives']:>5} {r['false_positives']:>5}")
        aucs.append(r['auc_roc'])

    mean_auc = np.mean(aucs) if aucs else 0
    min_auc = np.min(aucs) if aucs else 0
    max_auc = np.max(aucs) if aucs else 0
    std_auc = np.std(aucs) if aucs else 0

    print(f"\n  {'MOYENNE':<30} {mean_auc:>8.4f}")
    print(f"  {'MIN':<30} {min_auc:>8.4f}")
    print(f"  {'MAX':<30} {max_auc:>8.4f}")
    print(f"  {'STD':<30} {std_auc:>8.4f}")

    # Par type de défaut
    print(f"\n  Par type de défaut :")
    for ftype in ['Ball', 'Inner', 'Outer']:
        ft_aucs = [r['auc_roc'] for r in all_results if r['fault_type'] == ftype]
        if ft_aucs:
            print(f"    {ftype:<10} AUC={np.mean(ft_aucs):.4f} ± {np.std(ft_aucs):.4f}")

    # Par taille de défaut
    print(f"\n  Par taille de défaut :")
    for fsize in ['007', '014', '021']:
        fs_aucs = [r['auc_roc'] for r in all_results if r['fault_size'] == fsize]
        if fs_aucs:
            print(f"    {fsize}\"     AUC={np.mean(fs_aucs):.4f} ± {np.std(fs_aucs):.4f}")

    # Par charge
    print(f"\n  Par charge moteur :")
    for load in [0, 1, 2, 3]:
        ld_aucs = [r['auc_roc'] for r in all_results if r['load_hp'] == load]
        if ld_aucs:
            print(f"    {load}HP      AUC={np.mean(ld_aucs):.4f} ± {np.std(ld_aucs):.4f}")

    # --- Génération du rapport visuel ---
    _generate_multi_report(all_results, output_dir)

    # --- Sauvegarde JSON ---
    output = {
        'benchmark': 'CWRU Multi-Condition',
        'n_conditions': len(all_results),
        'summary': {
            'mean_auc': float(mean_auc),
            'min_auc': float(min_auc),
            'max_auc': float(max_auc),
            'std_auc': float(std_auc),
        },
        'conditions': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = os.path.join(output_dir, 'cwru_multi_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 JSON: {json_path}")
    print("\n✅ Benchmark multi-condition terminé.")


def _generate_multi_report(results, output_dir):
    """Heatmap AUC par (fault_type+size) × load."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("NATIVA — CWRU Multi-Condition AUC-ROC", fontsize=14, fontweight='bold')

    for idx, ftype in enumerate(['Ball', 'Inner', 'Outer']):
        ax = axes[idx]
        sizes = ['007', '014', '021']
        loads = [0, 1, 2, 3]

        grid = np.zeros((len(sizes), len(loads)))
        for r in results:
            if r['fault_type'] == ftype:
                si = sizes.index(r['fault_size'])
                li = loads.index(r['load_hp'])
                grid[si, li] = r['auc_roc']

        im = ax.imshow(grid, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(loads)))
        ax.set_xticklabels([f'{l}HP' for l in loads])
        ax.set_yticks(range(len(sizes)))
        ax.set_yticklabels([f'{s}"' for s in sizes])
        ax.set_title(f'{ftype} Fault')
        ax.set_xlabel('Load')
        ax.set_ylabel('Fault Size')

        for i in range(len(sizes)):
            for j in range(len(loads)):
                val = grid[i, j]
                color = 'white' if val < 0.75 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color)

    fig.colorbar(im, ax=axes, label='AUC-ROC', shrink=0.8)
    plt.tight_layout()

    path = os.path.join(output_dir, 'cwru_multi_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport multi-condition: {path}")


if __name__ == "__main__":
    main()
