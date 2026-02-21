"""
NATIVA-Learn — CWRU Bearing Fault Benchmark
=============================================
Benchmark honnête : NATIVA (SNN + Active Inference) vs Random Forest + FFT
Focus : Taux de Faux Négatifs (Recall) pour positionnement "Wake-Up Sensor"

Dataset : Case Western Reserve University Bearing Data Center
    - 12kHz Drive End, 4 conditions : Normal, Ball, Inner Race, Outer Race

Encodage SNN (état de l'art) :
    - Multi-scale frequency bands (8 bands, population coding)
    - Normalisation GLOBALE (calibrée sur les fenêtres normales)
    - Ref: MRA-SNN (2024), Balanced SNN for unsupervised AD

Usage:
    python benchmark_cwru.py
"""

import numpy as np
import os
import sys
import json
import ssl
import io
import urllib.request
import time
from pathlib import Path

from scipy.io import loadmat
from scipy.signal import stft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- NATIVA import (suppress debug prints) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nativa'))
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from nativa_network import NativaNetwork, NativaConfig
    from neuron import LIFNeuron
finally:
    sys.stdout = _real_stdout


# =====================================================================
# 1. CWRU DATASET
# =====================================================================

CWRU_FILES = {
    'normal': {
        'url': 'https://engineering.case.edu/sites/default/files/97.mat',
        'filename': '97.mat', 'key': 'X097_DE_time', 'label': 0
    },
    'ball': {
        'url': 'https://engineering.case.edu/sites/default/files/118.mat',
        'filename': '118.mat', 'key': 'X118_DE_time', 'label': 1
    },
    'inner': {
        'url': 'https://engineering.case.edu/sites/default/files/105.mat',
        'filename': '105.mat', 'key': 'X105_DE_time', 'label': 2
    },
    'outer': {
        'url': 'https://engineering.case.edu/sites/default/files/130.mat',
        'filename': '130.mat', 'key': 'X130_DE_time', 'label': 3
    },
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cwru')
SAMPLE_RATE = 12000  # Hz


def download_cwru():
    """Télécharge les fichiers CWRU si absents."""
    os.makedirs(DATA_DIR, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for name, info in CWRU_FILES.items():
        fpath = os.path.join(DATA_DIR, info['filename'])
        if os.path.exists(fpath):
            print(f"  ✅ {name}: déjà présent")
            continue
        print(f"  ⬇️  {name} ({info['filename']})...")
        try:
            req = urllib.request.Request(info['url'], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx) as resp, open(fpath, 'wb') as out:
                out.write(resp.read())
            print(f"  ✅ {name}: OK")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    print()


def load_cwru_data(window_size=1024, overlap=0.5):
    """Charge et segmente le dataset CWRU en fenêtres."""
    X_all, y_all = [], []
    step = int(window_size * (1 - overlap))

    for name, info in CWRU_FILES.items():
        fpath = os.path.join(DATA_DIR, info['filename'])
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Fichier manquant: {fpath}")

        mat = loadmat(fpath)
        key = info['key']
        if key not in mat:
            candidates = [k for k in mat.keys() if 'DE_time' in k]
            key = candidates[0] if candidates else None
            if not key:
                print(f"  ❌ Clé introuvable dans {info['filename']}")
                continue

        signal = mat[key].flatten()
        n_windows = (len(signal) - window_size) // step
        for i in range(n_windows):
            X_all.append(signal[i * step : i * step + window_size])
            y_all.append(info['label'])

    X = np.array(X_all, dtype=np.float64)
    y = np.array(y_all, dtype=int)
    y_binary = (y > 0).astype(int)

    for lbl, name in enumerate(['Normal', 'Ball', 'Inner', 'Outer']):
        print(f"   {name}: {np.sum(y == lbl)} fenêtres")
    print(f"   Total: {len(y)} | Sains: {np.sum(y_binary==0)} | Défauts: {np.sum(y_binary==1)}")

    return X, y, y_binary


# =====================================================================
# 2. MULTI-SCALE SPIKE ENCODER (état de l'art)
# =====================================================================

class MultiScaleSpikeEncoder:
    """
    Encodeur multi-échelle pour SNN — Approche état de l'art.

    Principe (réf: MRA-SNN 2024, population coding) :
        1. Découpe le spectre en N bandes de fréquence (multi-scale)
        2. Pour chaque pas de temps, calcule l'énergie par bande
        3. Normalise GLOBALEMENT (calibré sur données normales)
        4. Encode en spikes déterministes par seuil adaptatif

    Pourquoi ça marche :
        - Un roulement défectueux génère des harmoniques haute fréquence
        - L'énergie dans les bandes HF explose → plus de spikes HF
        - NATIVA détecte ce changement de pattern temporel via la Surprise

    Vérification :
        - normal_window → spike_train avec ~30-40% de densité, concentrée basses freq
        - fault_window  → spike_train avec ~50-70% de densité, répartie toutes freq
    """

    def __init__(self, n_bands=8, n_time_steps=64, sample_rate=12000):
        self.n_bands = n_bands
        self.n_time_steps = n_time_steps
        self.sample_rate = sample_rate
        self.global_max = None  # Calibré sur données normales

    def calibrate(self, normal_windows):
        """
        ÉTAPE CRITIQUE : Calcule le max global sur les fenêtres NORMALES.

        Pourquoi :
            Si on normalise par fenêtre → toute fenêtre ressemble à toute autre.
            Si on normalise globalement → une vibration 5x plus forte
            génère 5x plus de spikes. C'est cette différence que NATIVA détecte.

        Vérification :
            Après calibration, self.global_max devrait être un vecteur (n_bands,)
            avec des valeurs > 0. Les bandes basses doivent avoir un max plus élevé
            (le moteur sain vibre surtout en basses fréquences).
        """
        all_band_energies = []
        for window in normal_windows:
            bands = self._compute_bands(window)  # (n_time_steps, n_bands)
            all_band_energies.append(bands)

        all_energies = np.concatenate(all_band_energies, axis=0)  # (total_steps, n_bands)
        # Percentile 99 au lieu de max pour robustesse aux outliers
        self.global_max = np.percentile(all_energies, 99, axis=0)
        self.global_max = np.maximum(self.global_max, 1e-10)  # Éviter division par zéro

        print(f"   📐 Calibration: global_max par bande = {np.round(self.global_max, 4)}")

    def encode(self, window, threshold=0.15):
        """
        Encode une fenêtre vibratoire en spike train 2D.

        Args:
            window: np.ndarray (window_size,) — signal brut
            threshold: float — seuil d'activation (% du max calibré)

        Returns:
            spike_train: np.ndarray (n_time_steps, n_bands) — 0/1

        Vérification :
            - Densité fenêtre normale : ~20-40%
            - Densité fenêtre défaut :  ~50-80%
            - Si densités identiques → calibration ratée
        """
        bands = self._compute_bands(window)  # (n_time_steps, n_bands)

        # Normalisation GLOBALE (la clé de tout)
        normalized = bands / self.global_max[np.newaxis, :]

        # Encodage déterministe par seuil
        spike_train = (normalized > threshold).astype(np.float64)

        return spike_train

    def _compute_bands(self, window):
        """
        Décompose le signal en bandes de fréquence via STFT.

        Méthode : Short-Time Fourier Transform fenêtrée
            → matrice (freq_bins, time_steps)
            → regroupement en n_bands bandes d'énergie

        Ref: Standard ISO 10816 (bandes vibratoires industrielles)
        """
        nperseg = min(128, len(window) // 4)
        noverlap = nperseg // 2

        f, t, Zxx = stft(window, fs=self.sample_rate,
                         nperseg=nperseg, noverlap=noverlap,
                         return_onesided=True)

        # Magnitude au carré = énergie spectrale
        power = np.abs(Zxx) ** 2  # (n_freq, n_time)

        # Regrouper en n_bands bandes de fréquence (répartition linéaire)
        n_freq = power.shape[0]
        band_size = max(1, n_freq // self.n_bands)

        bands = np.zeros((power.shape[1], self.n_bands))
        for b in range(self.n_bands):
            start = b * band_size
            end = min((b + 1) * band_size, n_freq)
            if start < n_freq:
                bands[:, b] = np.mean(power[start:end, :], axis=0)

        # Sous-échantillonner/interpoler vers n_time_steps
        if bands.shape[0] != self.n_time_steps:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, bands.shape[0])
            x_new = np.linspace(0, 1, self.n_time_steps)
            interp = interp1d(x_old, bands, axis=0, kind='linear')
            bands = interp(x_new)

        return bands


# =====================================================================
# 3. FEATURE EXTRACTION (Random Forest baseline)
# =====================================================================

def extract_features(window):
    """Features classiques : temporelles + 10 FFT + 5 bandes d'énergie."""
    features = []
    features.append(np.sqrt(np.mean(window ** 2)))                          # RMS
    features.append(np.max(np.abs(window)))                                 # Peak
    features.append(features[-1] / (features[-2] + 1e-10))                  # Crest Factor
    centered = (window - np.mean(window)) / (np.std(window) + 1e-10)
    features.append(float(np.mean(centered ** 4)))                          # Kurtosis
    features.append(float(np.mean(centered ** 3)))                          # Skewness
    features.append(np.std(window))                                          # Std
    features.append(np.mean(np.abs(window)))                                # Mean Abs

    fft_vals = np.abs(np.fft.rfft(window))
    fft_norm = fft_vals / (np.max(fft_vals) + 1e-10)
    features.extend(fft_norm[:10].tolist())

    n_bands = 5
    bsz = len(fft_vals) // n_bands
    for b in range(n_bands):
        features.append(np.sum(fft_vals[b*bsz:(b+1)*bsz] ** 2))

    return np.array(features)


# =====================================================================
# 4. NATIVA BENCHMARK
# =====================================================================

def run_nativa_benchmark(X, y_binary, n_train_normal=100, verbose=True):
    """
    Pipeline NATIVA pour détection d'anomalies vibratoires.

    Architecture :
        1. Calibration de l'encodeur multi-échelle sur fenêtres NORMALES
        2. Calibration du réseau SNN (apprentissage STDP sur "normalité")
        3. Test : anomaly_score = mean(surprise) par fenêtre

    Vérification attendue :
        - Les scores normaux doivent être systématiquement plus bas
        - AUC-ROC > 0.7 minimum pour être utile
        - Idéalement > 0.85 pour être publishable
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  NATIVA SNN — ANOMALY DETECTION (Multi-Scale Encoding)")
        print("=" * 60)

    # --- Config réseau ---
    bearing_params = LIFNeuron(
        tau_m=50.0,       # Mémoire moyenne (écoute ~50ms de contexte)
        V_th=0.3,         # Seuil modéré (pas trop bas = pas trop de bruit)
        R_m=5.0,          # Résistance standard
        t_ref=2.0         # Période réfractaire (évite les rafales parasites)
    )

    cfg = NativaConfig(
        n_neurons=100,
        n_output_classes=2,
        neuron_params=bearing_params,
        use_kuramoto=True,
        kuramoto_coupling=2.0,
        weight_norm_target=100.0,
        use_adaptive_thresh=True,
        thresh_increment=0.5,
        seed=42
    )

    net = NativaNetwork(cfg)

    # --- Encodeur multi-échelle ---
    encoder = MultiScaleSpikeEncoder(n_bands=8, n_time_steps=64, sample_rate=SAMPLE_RATE)

    # --- Phase 1: Sélection des fenêtres normales ---
    normal_indices = np.where(y_binary == 0)[0]
    np.random.seed(42)
    train_indices = np.random.choice(
        normal_indices, size=min(n_train_normal, len(normal_indices)), replace=False
    )

    # --- Phase 2: Calibration de l'encodeur ---
    if verbose:
        print(f"\n🔧 Phase 1: Calibration encodeur sur {len(train_indices)} fenêtres normales...")
    encoder.calibrate(X[train_indices])

    # Diagnostic : vérifier que l'encodage sépare bien
    if verbose:
        # Prendre 1 fenêtre normale et 1 défaut pour vérifier
        sample_normal = encoder.encode(X[normal_indices[0]])
        fault_indices = np.where(y_binary == 1)[0]
        sample_fault = encoder.encode(X[fault_indices[0]])
        dens_normal = np.mean(sample_normal)
        dens_fault = np.mean(sample_fault)
        print(f"   🔍 Diagnostic encodage :")
        print(f"      Densité spikes NORMAL : {dens_normal:.3f}")
        print(f"      Densité spikes DÉFAUT : {dens_fault:.3f}")
        print(f"      Ratio défaut/normal   : {dens_fault / (dens_normal + 1e-10):.2f}x")
        if dens_fault / (dens_normal + 1e-10) < 1.2:
            print(f"      ⚠️  Ratio faible — l'encodage pourrait ne pas séparer assez")

    # --- Phase 3: Calibration réseau NATIVA ---
    if verbose:
        print(f"\n🧠 Phase 2: Apprentissage NATIVA sur {len(train_indices)} fenêtres normales...")
    t0 = time.time()

    for i, idx in enumerate(train_indices):
        spike_train = encoder.encode(X[idx])  # (64, 8) — 2D, bypass normalisation interne
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            net.feed(spike_train, mode="train")
        finally:
            sys.stdout = old_stdout
        if verbose and (i + 1) % 25 == 0:
            print(f"   Calibration: {i+1}/{len(train_indices)}")

    train_time = time.time() - t0
    if verbose:
        print(f"   ✅ Calibration terminée en {train_time:.1f}s")

    # --- Phase 4: Test ---
    test_indices = np.array([i for i in range(len(y_binary)) if i not in train_indices])
    if verbose:
        print(f"\n🔍 Phase 3: Test sur {len(test_indices)} fenêtres...")

    anomaly_scores = []
    t0 = time.time()

    for i, idx in enumerate(test_indices):
        spike_train = encoder.encode(X[idx])
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = net.feed(spike_train, mode="test")
        finally:
            sys.stdout = old_stdout

        surprise = np.array(result['surprise'])
        score = float(np.mean(surprise)) if len(surprise) > 0 else 0.0
        anomaly_scores.append(score)

        if verbose and (i + 1) % 250 == 0:
            print(f"   Test: {i+1}/{len(test_indices)}")

    test_time = time.time() - t0
    if verbose:
        print(f"   ✅ Test terminé en {test_time:.1f}s")

    anomaly_scores = np.array(anomaly_scores)
    y_test = y_binary[test_indices]

    # --- Diagnostic distributions ---
    if verbose:
        scores_normal = anomaly_scores[y_test == 0]
        scores_fault = anomaly_scores[y_test == 1]
        print(f"\n   📊 Distribution des scores de surprise :")
        print(f"      Normal : μ={np.mean(scores_normal):.6f}, σ={np.std(scores_normal):.6f}")
        print(f"      Défaut : μ={np.mean(scores_fault):.6f}, σ={np.std(scores_fault):.6f}")
        sep = (np.mean(scores_fault) - np.mean(scores_normal)) / (np.std(scores_normal) + 1e-10)
        print(f"      Séparation (d') : {sep:.2f} (>1.0 = bon, >2.0 = excellent)")

    return anomaly_scores, y_test, test_indices, train_time, test_time


# =====================================================================
# 5. THRESHOLD EVALUATION
# =====================================================================

def evaluate_thresholds(anomaly_scores, y_test, verbose=True):
    """Évalue NATIVA à différents seuils. Focus sur Recall (faux négatifs)."""
    if verbose:
        print("\n📏 Analyse des seuils de décision NATIVA:")

    results = {}

    # AUC-ROC
    try:
        auc = roc_auc_score(y_test, anomaly_scores)
    except ValueError:
        auc = 0.5
    results['auc_roc'] = auc
    if verbose:
        print(f"   AUC-ROC: {auc:.4f}")

    # Sweep des seuils
    percentiles = np.arange(5, 95, 2)
    best_f1, best_threshold = 0, None
    best_recall_threshold = None
    threshold_data = []

    for p in percentiles:
        threshold = np.percentile(anomaly_scores, p)
        y_pred = (anomaly_scores >= threshold).astype(int)
        if np.sum(y_pred) == 0 or np.sum(y_pred) == len(y_pred):
            continue

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        threshold_data.append({'percentile': p, 'threshold': float(threshold),
                               'precision': prec, 'recall': rec, 'f1': f1})

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        if rec >= 0.95 and (best_recall_threshold is None or threshold > best_recall_threshold):
            best_recall_threshold = threshold

    # Meilleur F1
    if best_threshold is not None:
        y_pred = (anomaly_scores >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        results['best_f1'] = {
            'threshold': float(best_threshold),
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        }
        if verbose:
            r = results['best_f1']
            print(f"\n   📊 Meilleur F1 (seuil={best_threshold:.6f}):")
            print(f"      Precision={r['precision']:.4f}  Recall={r['recall']:.4f}  F1={r['f1']:.4f}")
            print(f"      FN={r['false_negatives']}  FP={r['false_positives']}")

    # Wake-Up Mode (Recall ≥ 95%)
    if best_recall_threshold is not None:
        y_pred = (anomaly_scores >= best_recall_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        results['wake_up_mode'] = {
            'threshold': float(best_recall_threshold),
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        }
        if verbose:
            r = results['wake_up_mode']
            print(f"\n   🚨 Wake-Up Mode (Recall≥95%, seuil={best_recall_threshold:.6f}):")
            print(f"      Precision={r['precision']:.4f}  Recall={r['recall']:.4f}")
            print(f"      FN={r['false_negatives']}  FP={r['false_positives']}")

    results['threshold_curve'] = threshold_data
    return results


# =====================================================================
# 6. RANDOM FOREST BASELINE
# =====================================================================

def run_rf_benchmark(X, y_binary, verbose=True):
    """Random Forest + FFT features — baseline industrielle standard."""
    if verbose:
        print("\n" + "=" * 60)
        print("  RANDOM FOREST + FFT — BASELINE")
        print("=" * 60)

    t0 = time.time()
    X_feat = np.array([extract_features(w) for w in X])
    if verbose:
        print(f"   {X_feat.shape[1]} features extraites en {time.time()-t0:.1f}s")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_true, all_pred, all_proba = [], [], []

    t0 = time.time()
    for fold, (tr, te) in enumerate(skf.split(X_feat, y_binary)):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_feat[tr], y_binary[tr])
        all_true.extend(y_binary[te])
        all_pred.extend(clf.predict(X_feat[te]))
        all_proba.extend(clf.predict_proba(X_feat[te])[:, 1])
        if verbose:
            print(f"   Fold {fold+1}: Acc={accuracy_score(y_binary[te], clf.predict(X_feat[te])):.4f}")

    all_true, all_pred, all_proba = map(np.array, [all_true, all_pred, all_proba])
    cm = confusion_matrix(all_true, all_pred)

    results = {
        'accuracy': float(accuracy_score(all_true, all_pred)),
        'precision': float(precision_score(all_true, all_pred, zero_division=0)),
        'recall': float(recall_score(all_true, all_pred, zero_division=0)),
        'f1': float(f1_score(all_true, all_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(all_true, all_proba)),
        'confusion_matrix': cm.tolist(),
        'false_negatives': int(cm[1, 0]),
        'false_positives': int(cm[0, 1]),
        'train_time': time.time() - t0,
    }

    if verbose:
        print(f"\n   📊 RF: Acc={results['accuracy']:.4f} P={results['precision']:.4f} "
              f"R={results['recall']:.4f} F1={results['f1']:.4f} AUC={results['auc_roc']:.4f}")
        print(f"      FN={results['false_negatives']}  FP={results['false_positives']}")

    return results


# =====================================================================
# 7. REPORT
# =====================================================================

def generate_report(nativa_res, rf_res, output_dir):
    """Génère le rapport visuel comparatif."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NATIVA vs Random Forest — CWRU Bearing Fault Detection",
                 fontsize=14, fontweight='bold')

    # A. Bar chart comparatif
    ax = axes[0, 0]
    metrics = ['Precision', 'Recall', 'F1', 'AUC-ROC']
    nb = nativa_res.get('best_f1', {})
    nv = [nb.get('precision', 0), nb.get('recall', 0), nb.get('f1', 0), nativa_res.get('auc_roc', 0)]
    rv = [rf_res['precision'], rf_res['recall'], rf_res['f1'], rf_res['auc_roc']]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, nv, w, label='NATIVA', color='#2196F3', alpha=0.8)
    b2 = ax.bar(x + w/2, rv, w, label='Random Forest', color='#FF9800', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.15)
    ax.set_title("Comparaison Métriques"); ax.legend(); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(b1, nv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
                ha='center', fontsize=8)
    for bar, val in zip(b2, rv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
                ha='center', fontsize=8)

    # B-C. Confusion Matrices
    for idx, (title, res, cmap) in enumerate([
        ('NATIVA', nativa_res.get('best_f1', {}), 'Blues'),
        ('Random Forest', rf_res, 'Oranges')
    ]):
        ax = axes[idx // 2 + (1 if idx == 1 else 0), 1 if idx == 0 else 0]
        cm = np.array(res.get('confusion_matrix', [[0, 0], [0, 0]]))
        ax.imshow(cm, cmap=cmap, aspect='auto')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Sain', 'Défaut']); ax.set_yticklabels(['Sain', 'Défaut'])
        ax.set_xlabel('Prédit'); ax.set_ylabel('Réel'); ax.set_title(f'{title} — Confusion Matrix')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=14,
                        fontweight='bold', color='white' if cm[i][j] > cm.max()/2 else 'black')

    # D. Precision-Recall curve
    ax = axes[1, 1]
    if 'threshold_curve' in nativa_res and nativa_res['threshold_curve']:
        curve = nativa_res['threshold_curve']
        ax.plot([d['recall'] for d in curve], [d['precision'] for d in curve], 'b-o', ms=3)
        ax.axhline(y=rf_res['precision'], color='orange', ls='--', label=f"RF P={rf_res['precision']:.2f}")
        ax.axvline(x=rf_res['recall'], color='orange', ls=':', label=f"RF R={rf_res['recall']:.2f}")
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('NATIVA — Precision-Recall'); ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    wu = nativa_res.get('wake_up_mode')
    if wu:
        fig.text(0.5, 0.01,
                 f"🚨 Wake-Up Mode (Recall≥95%): P={wu['precision']:.2f} R={wu['recall']:.2f} "
                 f"FN={wu['false_negatives']} FP={wu['false_positives']}",
                 ha='center', fontsize=10, fontstyle='italic',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(output_dir, 'cwru_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Rapport: {path}")
    return path


# =====================================================================
# 8. MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("  NATIVA-Learn — CWRU BEARING FAULT BENCHMARK v2")
    print("  Multi-Scale Encoding + Global Normalization")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    # 1. Download
    print("\n📥 Étape 1: Dataset CWRU...")
    download_cwru()

    # 2. Load
    print("📦 Étape 2: Chargement...")
    X, y, y_binary = load_cwru_data(window_size=1024, overlap=0.5)

    # 3. RF baseline
    print("\n🌲 Étape 3: Random Forest baseline...")
    rf_results = run_rf_benchmark(X, y_binary)

    # 4. NATIVA
    print("\n🧠 Étape 4: NATIVA SNN...")
    scores, y_test, test_idx, t_train, t_test = run_nativa_benchmark(X, y_binary, n_train_normal=100)
    nativa_results = evaluate_thresholds(scores, y_test)
    nativa_results['train_time'] = t_train
    nativa_results['test_time'] = t_test

    # 5. Verdict
    print("\n" + "=" * 60)
    print("  📋 VERDICT FINAL")
    print("=" * 60)

    nb = nativa_results.get('best_f1', {})
    print(f"\n  {'Métrique':<15} {'NATIVA':>10} {'RF':>10} {'Delta':>10}")
    print(f"  {'-'*47}")
    for m in ['precision', 'recall', 'f1']:
        nv = nb.get(m, 0)
        rv = rf_results[m]
        d = nv - rv
        print(f"  {m.capitalize():<15} {nv:>10.4f} {rv:>10.4f} {d:>+10.4f} {'↑' if d > 0 else '↓'}")
    nv = nativa_results.get('auc_roc', 0)
    rv = rf_results['auc_roc']
    print(f"  {'AUC-ROC':<15} {nv:>10.4f} {rv:>10.4f} {nv-rv:>+10.4f} {'↑' if nv > rv else '↓'}")

    fn_n = nb.get('false_negatives', '?')
    print(f"\n  Faux Négatifs: NATIVA={fn_n}  RF={rf_results['false_negatives']}")

    wu = nativa_results.get('wake_up_mode')
    if wu:
        print(f"\n  🚨 Wake-Up Sensor (Recall≥95%): P={wu['precision']:.4f} R={wu['recall']:.4f}")
        print(f"     FN={wu['false_negatives']}  FP={wu['false_positives']}")

    # 6. Report
    generate_report(nativa_results, rf_results, output_dir)

    # 7. JSON
    all_results = {
        'dataset': 'CWRU Bearing (12k DE, 0.007in)', 'window_size': 1024,
        'encoding': 'MultiScale 8 bands, 64 timesteps, global norm',
        'nativa': nativa_results, 'random_forest': rf_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = os.path.join(output_dir, 'cwru_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"💾 JSON: {json_path}")
    print("\n✅ Benchmark terminé.")


if __name__ == "__main__":
    main()
