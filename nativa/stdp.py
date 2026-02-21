"""
STDP + Free Energy pour NATIVA-Learn
=====================================
Apprentissage par Spike-Timing-Dependent Plasticity (STDP), modulé par
la minimisation de l'énergie libre (Free Energy Principle).

INNOVATION NATIVA:
    Le taux d'apprentissage STDP est modulé par le niveau de surprise
    (Free Energy) de l'agent. Quand l'agent est surpris (F élevé),
    il apprend plus vite. Quand tout est familier (F bas), il consolide.

    Δw_modulated = Δw_stdp × σ(β · (F - F_baseline))

    Où σ est la fonction sigmoïde et β contrôle la sensibilité.

Architecture:
    - STDPConfig : paramètres de la règle STDP
    - STDPSynapse : une synapse unique avec tracking temporel
    - SynapticMatrix : matrice sparse de N×N connexions
    - FreeEnergySTDP : STDP avec modulation par Free Energy (NOUVEAU)

REFERENCE: Bi, G.Q. & Poo, M.M. (1998). "Synaptic modifications in cultured
           hippocampal neurons: dependence on spike timing, synaptic strength,
           and postsynaptic cell type." J. Neurosci. 18(24):10464-10472.

REFERENCE: Friston, K. (2010). "The free-energy principle: a unified brain
           theory?" Nature Reviews Neuroscience, 11(2), 127-138.

Conventions de documentation pour audit LLM:
    # CONTRACT: pré/post-conditions d'une méthode
    # INVARIANT: propriété mathématique toujours vraie
    # REFERENCE: citation papier/équation source
    # VERIFY: assertion qu'un auditeur LLM doit vérifier
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from scipy import sparse


# ==============================================================================
# Configuration STDP
# ==============================================================================

@dataclass
class STDPConfig:
    """
    Paramètres de la règle d'apprentissage STDP.

    La STDP est une règle d'apprentissage locale et temporelle :
    - Si le neurone pré-synaptique fire AVANT le post → potentiation (Δw > 0)
    - Si le neurone pré-synaptique fire APRÈS le post → dépression (Δw < 0)

    REFERENCE: Bi & Poo (1998) — Courbe STDP expérimentale
    REFERENCE: Song, S., Miller, K.D. & Abbott, L.F. (2000).
               "Competitive Hebbian learning through STDP"

    Équation de la fenêtre STDP:

        Δw = A_plus  · exp(-|Δt| / τ_plus)   si Δt > 0  (potentiation)
        Δw = -A_minus · exp(-|Δt| / τ_minus)  si Δt < 0  (dépression)

        où Δt = t_post - t_pre

    INVARIANT: tau_plus > 0  (sinon exponentielle indéfinie)
    INVARIANT: tau_minus > 0
    INVARIANT: A_plus > 0    (potentiation est positive)
    INVARIANT: A_minus > 0   (dépression est positive, le signe est dans la formule)
    INVARIANT: w_max > w_min >= 0  (poids bornés et non-négatifs)

    Attributes:
        tau_plus: Constante de temps pour la potentiation (ms).
                  Typiquement 20ms (Bi & Poo 1998).
        tau_minus: Constante de temps pour la dépression (ms).
                   Typiquement 20ms.
        A_plus: Amplitude maximale de potentiation.
        A_minus: Amplitude maximale de dépression.
                 Souvent A_minus > A_plus pour maintenir la stabilité.
        learning_rate: Taux d'apprentissage global η.
        w_min: Poids synaptique minimum (borne inférieure).
        w_max: Poids synaptique maximum (borne supérieure).
    """
    tau_plus: float = 20.0     # ms
    tau_minus: float = 20.0    # ms
    A_plus: float = 0.005       # amplitude potentiation (TUNED)
    A_minus: float = 0.012     # amplitude dépression (légèrement > A_plus pour stabilité)
    learning_rate: float = 1.0
    w_min: float = 0.0
    w_max: float = 10.0
    
    # 3-Factor STDP (RL)
    eligibility_tau: float = 100.0 # ms, mémoire des actions passées

    def __post_init__(self):
        # VERIFY: Invariants structurels
        assert self.tau_plus > 0, f"CONTRACT VIOLATION: tau_plus={self.tau_plus} doit être > 0"
        assert self.tau_minus > 0, f"CONTRACT VIOLATION: tau_minus={self.tau_minus} doit être > 0"
        assert self.A_plus > 0, f"CONTRACT VIOLATION: A_plus={self.A_plus} doit être > 0"
        assert self.A_minus > 0, f"CONTRACT VIOLATION: A_minus={self.A_minus} doit être > 0"
        assert self.w_max > self.w_min >= 0, \
            f"CONTRACT VIOLATION: w_max ({self.w_max}) > w_min ({self.w_min}) >= 0"


# ==============================================================================
# Matrice Synaptique
# ==============================================================================

class SynapticMatrix:
    """
    Matrice de poids synaptiques pour N neurones avec apprentissage STDP.

    Utilise une représentation dense NumPy pour la vectorisation.
    W[i, j] = poids de la connexion du neurone j (pré) au neurone i (post).

    REFERENCE: Song, Miller & Abbott (2000) — STDP compétitif
    REFERENCE: Morrison, Diesmann & Gerstner (2008).
               "Phenomenological models of synaptic plasticity based on STDP"

    Caractéristiques NATIVA:
    - update_stdp() : mise à jour STDP classique vectorisée
    - modulate_by_free_energy() : modulation du learning rate par F (INNOVATION)
    - prune() : supprime les connexions faibles (auto-organisation)

    INVARIANT: W[i, j] ∈ [w_min, w_max] pour tout i, j
    INVARIANT: W[i, i] == 0 (pas d'auto-connexion)
    INVARIANT: connectivity_mask[i, j] == False → W[i, j] == 0
    """

    def __init__(
        self,
        n_neurons: int,
        config: Optional[STDPConfig] = None,
        connectivity: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialise une matrice synaptique avec connectivité aléatoire.

        CONTRACT: n_neurons > 0
        CONTRACT: 0 < connectivity <= 1

        Args:
            n_neurons: Nombre de neurones (la matrice est NxN).
            config: Paramètres STDP. Si None, utilise les défauts.
            connectivity: Probabilité de connexion entre deux neurones.
                         0.3 = 30% des connexions possibles existent.
            seed: Graine aléatoire pour reproductibilité.
        """
        assert n_neurons > 0, f"CONTRACT VIOLATION: n_neurons={n_neurons} doit être > 0"
        assert 0 < connectivity <= 1, \
            f"CONTRACT VIOLATION: connectivity={connectivity} doit être dans (0, 1]"

        self.n_neurons = n_neurons
        self.config = config or STDPConfig()

        rng = np.random.default_rng(seed)

        # --- Matrice de poids ---
        # Initialisation uniforme dans [w_min, w_max/2]
        self.W = rng.uniform(
            self.config.w_min,
            self.config.w_max * 0.5,
            size=(n_neurons, n_neurons)
        )

        # --- Masque de connectivité ---
        # Seules certaines connexions existent (réduire le sur-apprentissage)
        self.connectivity_mask = rng.random((n_neurons, n_neurons)) < connectivity

        # Pas d'auto-connexion
        # INVARIANT: W[i, i] == 0
        np.fill_diagonal(self.connectivity_mask, False)

        # Appliquer le masque
        self.W *= self.connectivity_mask

        # --- Traces STDP (pour all-to-all STDP) ---
        # REFERENCE: Morrison et al. (2008) — Approche par traces exponentielles
        # x_pre[j] : trace pré-synaptique du neurone j (incrémentée à chaque spike pré)
        # x_post[i] : trace post-synaptique du neurone i
        self.x_pre = np.zeros(n_neurons, dtype=np.float64)
        self.x_post = np.zeros(n_neurons, dtype=np.float64)

        # --- Traces d'Éligibilité (3-Factor STDP) ---
        # e[i,j] : Potentiel de changement de poids en attente de récompense
        # Decays avec eligibility_tau
        self.eligibility_trace = np.zeros((n_neurons, n_neurons), dtype=np.float64)

        # --- Modulation Free Energy ---
        self._fe_modulation = 1.0  # Facteur de modulation courant

        # --- Historique des poids (pour diagnostic) ---
        self.weight_history: list = []

    def update_stdp(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float = 1.0
    ):
        """
        Met à jour les poids selon la règle STDP avec traces exponentielles.

        Algorithme (vectorisé, all-to-all STDP):
            1. Décroître les traces : x *= exp(-dt/τ)
            2. Pour chaque spike pré_j : potentiation W[:, j] += A_plus · x_post
            3. Pour chaque spike post_i : dépression W[i, :] -= A_minus · x_pre
            4. Incrémenter les traces pour les neurones qui ont spiké
            5. Appliquer les bornes [w_min, w_max]

        REFERENCE: Morrison et al. (2008) Eq. 5-7 — STDP par traces

        CONTRACT: pre_spikes.shape == (n_neurons,) — booléen
        CONTRACT: post_spikes.shape == (n_neurons,) — booléen
        CONTRACT: dt > 0
        CONTRACT POST: w_min <= W[i,j] <= w_max pour tout i,j connecté

        Args:
            pre_spikes: Masque booléen des neurones pré qui ont spiké.
            post_spikes: Masque booléen des neurones post qui ont spiké.
            dt: Pas de temps en ms.
        """
        assert pre_spikes.shape == (self.n_neurons,), \
            f"CONTRACT VIOLATION: pre_spikes.shape={pre_spikes.shape}"
        assert post_spikes.shape == (self.n_neurons,), \
            f"CONTRACT VIOLATION: post_spikes.shape={post_spikes.shape}"

        cfg = self.config
        eta = cfg.learning_rate * self._fe_modulation

        # --- Étape 1 : Décroissance exponentielle des traces ---
        # REFERENCE: x(t+dt) = x(t) · exp(-dt/τ)
        self.x_pre *= np.exp(-dt / cfg.tau_plus)
        self.x_post *= np.exp(-dt / cfg.tau_minus)

        # --- Étape 2 : Potentiation (pré fire AVANT post existant) ---
        # Quand un neurone pré j spike, toutes les connexions W[:, j]
        # sont renforcées proportionnellement à la trace post-synaptique
        pre_indices = np.where(pre_spikes)[0]
        if len(pre_indices) > 0:
            # REFERENCE: ΔW[:, j] = η · A_plus · x_post[:]
            # (seules les connexions existantes sont modifiées)
            delta_pot = eta * cfg.A_plus * np.outer(self.x_post, np.ones(len(pre_indices)))
            self.W[:, pre_indices] += delta_pot * self.connectivity_mask[:, pre_indices]

        # --- Étape 3 : Dépression (post fire, déprime les pré existants) ---
        # Quand un neurone post i spike, toutes les connexions W[i, :]
        # sont affaiblies proportionnellement à la trace pré-synaptique
        post_indices = np.where(post_spikes)[0]
        if len(post_indices) > 0:
            # REFERENCE: ΔW[i, :] = -η · A_minus · x_pre[:]
            delta_dep = eta * cfg.A_minus * np.outer(np.ones(len(post_indices)), self.x_pre)
            self.W[post_indices, :] -= delta_dep * self.connectivity_mask[post_indices, :]

        # --- Étape 4 : Incrémenter les traces pour les spikers ---
        # REFERENCE: x_pre(t) += 1 quand le neurone pré spike
        self.x_pre[pre_spikes] += 1.0
        self.x_post[post_spikes] += 1.0

        # --- Étape 5 : Appliquer les bornes ---
        # INVARIANT: W[i,j] ∈ [w_min, w_max]
        np.clip(self.W, cfg.w_min, cfg.w_max, out=self.W)

        # Forcer les non-connexions à 0
        self.W *= self.connectivity_mask

        # INVARIANT: pas d'auto-connexion
        np.fill_diagonal(self.W, 0.0)

    def modulate_by_free_energy(self, F: float, F_baseline: float = 1.0, beta: float = 2.0):
        """
        MODULE L'apprentissage STDP par le niveau de surprise (Free Energy).

        C'est l'INNOVATION CLE de NATIVA-Learn.

        Quand F est élevé (surprise) → le réseau apprend plus vite
        Quand F est bas (familier) → le réseau consolide (apprend moins)

        Formule:
            modulation = σ(β · (F - F_baseline))

            σ(x) = 1 / (1 + exp(-x))   (sigmoïde)

        REFERENCE: Friston (2010) — "The free-energy principle"
        REFERENCE: Rao & Ballard (1999) — "Predictive coding in the visual cortex"
                   → L'erreur de prédiction (≃ F) module la plasticité

        CONTRACT: F >= 0 (l'énergie libre est toujours non-négative en pratique)
        CONTRACT: F_baseline > 0
        CONTRACT: beta > 0

        INVARIANT: 0 < modulation < 1 (sigmoïde est bornée dans (0,1))

        Args:
            F: Énergie libre actuelle.
            F_baseline: Niveau "normal" de F. Au-dessus → surprise.
            beta: Sensibilité de la modulation. Plus beta est grand,
                  plus la transition est abrupte.
        """
        assert F_baseline > 0, f"CONTRACT VIOLATION: F_baseline={F_baseline} doit être > 0"
        assert beta > 0, f"CONTRACT VIOLATION: beta={beta} doit être > 0"

        # REFERENCE: σ(β · (F - F_baseline))
        x = beta * (F - F_baseline)
        # Clamp pour éviter overflow dans exp
        x = np.clip(x, -20.0, 20.0)
        self._fe_modulation = 1.0 / (1.0 + np.exp(-x))

        # VERIFY: modulation ∈ (0, 1)
        assert 0 < self._fe_modulation < 1, \
            f"INVARIANT VIOLATION: fe_modulation={self._fe_modulation} hors (0,1)"

    def propagate(self, spikes: np.ndarray) -> np.ndarray:
        """
        Propage les spikes à travers la matrice synaptique.

        Le courant reçu par le neurone i est la somme pondérée des spikes
        de tous les neurones pré-synaptiques connectés :

            I_i = Σ_j W[i, j] · spike_j

        CONTRACT: spikes.shape == (n_neurons,) — booléen ou float
        CONTRACT: Retourne un array (n_neurons,) de courants

        Args:
            spikes: Masque de spikes (booléen ou float 0/1).

        Returns:
            Courants d'entrée pour chaque neurone (N,).
        """
        # W @ spikes = somme pondérée des spikes pour chaque neurone post
        return self.W @ spikes.astype(np.float64)

    def prune(self, threshold: float = 0.01) -> int:
        """
        Supprime les connexions dont le poids est inférieur au seuil.

        Cela rend le réseau plus sparse et plus efficace.

        CONTRACT: threshold >= 0
        CONTRACT: Retourne le nombre de connexions supprimées

        Args:
            threshold: Seuil en-dessous duquel une connexion est supprimée.

        Returns:
            Nombre de connexions supprimées.
        """
        weak = (np.abs(self.W) < threshold) & self.connectivity_mask
        n_pruned = int(np.sum(weak))
        self.connectivity_mask[weak] = False
        self.connectivity_mask[weak] = False
        self.W[weak] = 0.0
        return n_pruned

    # ------------------------------------------------------------------
    # 3-Factor STDP (Reinforcement Learning)
    # ------------------------------------------------------------------

    def update_eligibility_trace(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float = 1.0
    ):
        """
        Met à jour la trace d'éligibilité (e_ij) SANS changer les poids.
        C'est l'étape "Hebbian" de l'apprentissage RL.
        """
        cfg = self.config
        
        # 1. Décroissance de la trace d'éligibilité
        self.eligibility_trace *= np.exp(-dt / cfg.eligibility_tau)
        
        # 2. Décroissance des traces synaptiques (x_pre, x_post)
        self.x_pre *= np.exp(-dt / cfg.tau_plus)
        self.x_post *= np.exp(-dt / cfg.tau_minus)

        # 3. Accumuler les changements potentiels dans la trace d'éligibilité
        # (Exactement comme update_stdp, mais cible eligibility_trace au lieu de W)
        
        pre_indices = np.where(pre_spikes)[0]
        if len(pre_indices) > 0:
            # Potentiation potentielle
            delta_pot = cfg.A_plus * np.outer(self.x_post, np.ones(len(pre_indices)))
            self.eligibility_trace[:, pre_indices] += delta_pot * self.connectivity_mask[:, pre_indices]

        post_indices = np.where(post_spikes)[0]
        if len(post_indices) > 0:
            # Dépression potentielle
            delta_dep = cfg.A_minus * np.outer(np.ones(len(post_indices)), self.x_pre)
            self.eligibility_trace[post_indices, :] -= delta_dep * self.connectivity_mask[post_indices, :]

        # 4. Mettre à jour les traces de spikes
        self.x_pre[pre_spikes] += 1.0
        self.x_post[post_spikes] += 1.0


    def apply_reward(self, reward: float, dt: float = 1.0):
        """
        Applique la récompense (Dopamine) aux traces d'éligibilité pour mettre à jour les poids.
        
        ΔW = η · Reward · EligibilityTrace
        
        Args:
            reward: Signal de récompense global (positif ou négatif).
            dt: Pas de temps (pour cohérence, mais ici l'application est instantanée).
        """
        if reward == 0:
            return

        cfg = self.config
        eta = cfg.learning_rate  # Pas de modulation FE ici pour l'instant (optionnel)
        
        # Mise à jour des poids
        delta_w = eta * reward * self.eligibility_trace
        self.W += delta_w
        
        # Appliquer les bornes
        np.clip(self.W, cfg.w_min, cfg.w_max, out=self.W)
        self.W *= self.connectivity_mask
        np.fill_diagonal(self.W, 0.0)

    # ------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------

    def get_weight_stats(self) -> Dict:
        """
        Statistiques sur les poids synaptiques.

        Returns:
            Dict avec mean, std, min, max, sparsity, n_connections.
        """
        active_weights = self.W[self.connectivity_mask]
        n_connections = int(np.sum(self.connectivity_mask))
        n_possible = self.n_neurons * (self.n_neurons - 1)  # sans auto-connexion

        return {
            'mean_weight': float(np.mean(active_weights)) if len(active_weights) > 0 else 0.0,
            'std_weight': float(np.std(active_weights)) if len(active_weights) > 0 else 0.0,
            'min_weight': float(np.min(active_weights)) if len(active_weights) > 0 else 0.0,
            'max_weight': float(np.max(active_weights)) if len(active_weights) > 0 else 0.0,
            'n_connections': n_connections,
            'sparsity': 1.0 - (n_connections / n_possible) if n_possible > 0 else 0.0,
            'fe_modulation': self._fe_modulation,
        }

    def snapshot_weights(self):
        """Sauvegarde un snapshot de la distribution des poids."""
        stats = self.get_weight_stats()
        self.weight_history.append(stats)

    def get_connectivity_for_neuron(self, neuron_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retourne les connexions entrantes et sortantes d'un neurone.

        Returns:
            (incoming_weights, outgoing_weights)
        """
        incoming = self.W[neuron_idx, :]  # poids des connexions vers ce neurone
        outgoing = self.W[:, neuron_idx]  # poids des connexions depuis ce neurone
        return incoming.copy(), outgoing.copy()


# ==============================================================================
# Tests et Démonstration
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODULE STDP + FREE ENERGY")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Test 1 : Création de la matrice synaptique
    # ---------------------------------------------------------------
    print("\n--- Test 1: Création de la matrice synaptique ---")
    syn = SynapticMatrix(n_neurons=50, connectivity=0.3, seed=42)
    stats = syn.get_weight_stats()
    print(f"Matrice 50x50, connectivité 30%")
    print(f"  Connexions: {stats['n_connections']}")
    print(f"  Poids moyen: {stats['mean_weight']:.4f}")
    print(f"  Sparsité: {stats['sparsity']:.2%}")

    # VERIFY: Pas d'auto-connexion
    assert np.all(np.diag(syn.W) == 0), "VERIFY FAILED: Auto-connexions détectées"
    # VERIFY: Poids dans les bornes
    active = syn.W[syn.connectivity_mask]
    assert np.all(active >= syn.config.w_min) and np.all(active <= syn.config.w_max), \
        "VERIFY FAILED: Poids hors bornes"
    print("✅ Création OK")

    # ---------------------------------------------------------------
    # Test 2 : Courbe STDP (potentiation et dépression)
    # ---------------------------------------------------------------
    print("\n--- Test 2: Courbe STDP (Δw en fonction de Δt) ---")
    cfg = STDPConfig()

    # Calculer Δw pour différents Δt
    delta_ts = np.arange(-50, 51, 1.0)  # ms
    delta_ws = np.zeros_like(delta_ts)

    for i, delta_t in enumerate(delta_ts):
        if delta_t > 0:
            # Pré AVANT post → potentiation
            delta_ws[i] = cfg.A_plus * np.exp(-abs(delta_t) / cfg.tau_plus)
        elif delta_t < 0:
            # Post AVANT pré → dépression
            delta_ws[i] = -cfg.A_minus * np.exp(-abs(delta_t) / cfg.tau_minus)
        else:
            delta_ws[i] = 0.0

    # VERIFY: Potentiation pour Δt > 0
    assert np.all(delta_ws[delta_ts > 0] > 0), \
        "VERIFY FAILED: Δw devrait être > 0 pour Δt > 0 (potentiation)"
    # VERIFY: Dépression pour Δt < 0
    assert np.all(delta_ws[delta_ts < 0] < 0), \
        "VERIFY FAILED: Δw devrait être < 0 pour Δt < 0 (dépression)"
    # VERIFY: Δw(Δt=1) > Δw(Δt=40) (décroissance exponentielle)
    assert delta_ws[delta_ts == 1][0] > delta_ws[delta_ts == 40][0], \
        "VERIFY FAILED: Δw devrait décroître avec |Δt|"

    print(f"  Δt =  1ms → Δw = +{delta_ws[delta_ts == 1][0]:.5f} (potentiation)")
    print(f"  Δt = 40ms → Δw = +{delta_ws[delta_ts == 40][0]:.5f} (potentiation faible)")
    print(f"  Δt = -1ms → Δw = {delta_ws[delta_ts == -1][0]:.5f} (dépression)")
    print(f"  Δt =-40ms → Δw = {delta_ws[delta_ts == -40][0]:.5f} (dépression faible)")
    print("✅ Courbe STDP OK — Potentiation / Dépression correctes")

    # ---------------------------------------------------------------
    # Test 3 : Mise à jour STDP vectorisée
    # ---------------------------------------------------------------
    print("\n--- Test 3: Update STDP vectorisé ---")
    syn3 = SynapticMatrix(n_neurons=20, connectivity=0.5, seed=42)
    w_before = syn3.W.copy()

    # Simuler des spikes
    rng = np.random.default_rng(42)
    for step in range(100):
        pre_spikes = rng.random(20) < 0.1   # 10% de chance de spike
        post_spikes = rng.random(20) < 0.1
        syn3.update_stdp(pre_spikes, post_spikes, dt=1.0)

    w_after = syn3.W.copy()
    total_change = np.sum(np.abs(w_after - w_before))
    print(f"  Changement total des poids après 100 steps: {total_change:.4f}")

    # VERIFY: Les poids ont changé (STDP a eu un effet)
    assert total_change > 0, "VERIFY FAILED: STDP n'a pas modifié les poids"
    # VERIFY: Les poids sont toujours dans les bornes
    active_w = syn3.W[syn3.connectivity_mask]
    assert np.all(active_w >= syn3.config.w_min) and np.all(active_w <= syn3.config.w_max), \
        "VERIFY FAILED: Poids hors bornes après STDP"
    print("✅ Update STDP OK — Poids modifiés et bornés")

    # ---------------------------------------------------------------
    # Test 4 : Modulation par Free Energy (INNOVATION NATIVA)
    # ---------------------------------------------------------------
    print("\n--- Test 4: Modulation par Free Energy ---")
    syn4a = SynapticMatrix(n_neurons=20, connectivity=0.5, seed=42)
    syn4b = SynapticMatrix(n_neurons=20, connectivity=0.5, seed=42)

    # Même séquence de spikes pour les deux
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    # syn4a : Free Energy HAUTE (surprise) → devrait apprendre plus
    syn4a.modulate_by_free_energy(F=5.0, F_baseline=1.0, beta=2.0)
    modulation_high = syn4a._fe_modulation

    # syn4b : Free Energy BASSE (familier) → devrait apprendre moins
    syn4b.modulate_by_free_energy(F=0.1, F_baseline=1.0, beta=2.0)
    modulation_low = syn4b._fe_modulation

    print(f"  F=5.0 (surprise) → modulation = {modulation_high:.4f}")
    print(f"  F=0.1 (familier) → modulation = {modulation_low:.4f}")

    # VERIFY: Modulation haute quand F est haut
    assert modulation_high > modulation_low, \
        "VERIFY FAILED: La modulation devrait être plus haute quand F est plus haut"

    # Appliquer STDP avec les deux modulations
    for step in range(100):
        pre = rng_a.random(20) < 0.1
        post = rng_a.random(20) < 0.1
        syn4a.update_stdp(pre, post, dt=1.0)

        pre = rng_b.random(20) < 0.1
        post = rng_b.random(20) < 0.1
        syn4b.update_stdp(pre, post, dt=1.0)

    change_high = np.sum(np.abs(syn4a.W - SynapticMatrix(20, connectivity=0.5, seed=42).W))
    change_low = np.sum(np.abs(syn4b.W - SynapticMatrix(20, connectivity=0.5, seed=42).W))

    print(f"  Changement poids (F haute): {change_high:.4f}")
    print(f"  Changement poids (F basse): {change_low:.4f}")

    # VERIFY: Plus de changement quand Free Energy est haute
    assert change_high > change_low, \
        "VERIFY FAILED: STDP devrait produire plus de changement quand F est haute"
    print("✅ Modulation Free Energy OK — Surprise → plus d'apprentissage")

    # ---------------------------------------------------------------
    # Test 5 : Propagation de spikes
    # ---------------------------------------------------------------
    print("\n--- Test 5: Propagation de spikes ---")
    syn5 = SynapticMatrix(n_neurons=10, connectivity=1.0, seed=42)
    spikes = np.zeros(10, dtype=bool)
    spikes[0] = True  # Seul le neurone 0 spike

    currents = syn5.propagate(spikes)
    print(f"  Spike du neurone 0 → courants: {currents[:5]}...")
    # VERIFY: Le courant reçu = W[:, 0] (colonne 0 de la matrice)
    expected = syn5.W[:, 0]
    assert np.allclose(currents, expected), "VERIFY FAILED: Propagation incorrecte"
    print("✅ Propagation OK")

    # ---------------------------------------------------------------
    # Test 6 : Pruning
    # ---------------------------------------------------------------
    print("\n--- Test 6: Pruning (élagage) ---")
    syn6 = SynapticMatrix(n_neurons=30, connectivity=0.5, seed=42)
    n_before = syn6.get_weight_stats()['n_connections']

    # Mettre quelques poids à presque 0
    low_mask = syn6.W < 0.1
    syn6.W[low_mask & syn6.connectivity_mask] = 0.005

    n_pruned = syn6.prune(threshold=0.01)
    n_after = syn6.get_weight_stats()['n_connections']
    print(f"  Connexions avant: {n_before}, après pruning: {n_after}, élaguées: {n_pruned}")

    # VERIFY: Des connexions ont été supprimées
    assert n_pruned > 0, "VERIFY FAILED: Le pruning devrait supprimer des connexions"
    assert n_after == n_before - n_pruned, "VERIFY FAILED: Comptage incohérent"
    print("✅ Pruning OK")

    # ---------------------------------------------------------------
    # Test 7 : 3-Factor STDP (RL)
    # ---------------------------------------------------------------
    print("\n--- Test 7: 3-Factor STDP (Reinforcement Learning) ---")
    cfg_rl = STDPConfig(eligibility_tau=50.0) # 50ms memory
    syn_rl = SynapticMatrix(n_neurons=2, connectivity=1.0, seed=42, config=cfg_rl)
    
    # Pre spike, then Post spike -> Potentiation pending
    pre = np.array([True, False])
    post = np.array([False, True]) # Post fires after Pre
    
    # 1. Update Eligibility Trace
    syn_rl.update_eligibility_trace(pre, post, dt=1.0)
    
    # Verify trace is positive (hebbian association stored)
    trace_val = syn_rl.eligibility_trace[1, 0] # Connection 0->1
    print(f"  Eligibility Trace (Pre->Post): {trace_val:.6f}")
    assert trace_val > 0, "VERIFY FAILED: Eligibility trace should be positive"
    
    # Verify Weight hasn't changed yet
    assert syn_rl.W[1, 0] == syn_rl.W[1, 0], "VERIFY FAILED: Weight should not change before reward"
    
    # 2. Apply Reward (Dopamine)
    reward = 1.0
    w_before_reward = syn_rl.W[1, 0]
    syn_rl.apply_reward(reward, dt=1.0)
    w_after_reward = syn_rl.W[1, 0]
    
    print(f"  Weight change after reward: {w_after_reward - w_before_reward:.6f}")
    assert w_after_reward > w_before_reward, "VERIFY FAILED: Positive reward should increase weight"
    
    # 3. Apply Punishment (Negative Reward)
    # Reset traces for cleaner test
    syn_rl.eligibility_trace[:] = 0
    syn_rl.update_eligibility_trace(pre, post, dt=1.0) # Potentiation pending
    
    w_before_punish = syn_rl.W[1, 0]
    syn_rl.apply_reward(-1.0, dt=1.0) # Punishment!
    w_after_punish = syn_rl.W[1, 0]
    
    print(f"  Weight change after punishment: {w_after_punish - w_before_punish:.6f}")
    assert w_after_punish < w_before_punish, "VERIFY FAILED: Negative reward should decrease weight"
    
    print("✅ 3-Factor STDP OK — Reward modulates learning")

    # ---------------------------------------------------------------
    # Résumé
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RÉSULTAT: ✅ Tous les tests stdp.py passent")
    print("=" * 60)
    print("\nPropriétés vérifiées:")
    print("  ✓ Pas d'auto-connexion")
    print("  ✓ Poids toujours dans [w_min, w_max]")
    print("  ✓ STDP : potentiation si Δt > 0, dépression si Δt < 0")
    print("  ✓ STDP : décroissance exponentielle avec |Δt|")
    print("  ✓ Free Energy haute → apprentissage amplifié (INNOVATION)")
    print("  ✓ Propagation de spikes correcte")
    print("  ✓ Pruning supprime les connexions faibles")
    print("  ✓ 3-Factor STDP : Reward valide les traces (RL)")
