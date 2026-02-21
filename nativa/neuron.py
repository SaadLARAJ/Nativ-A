"""
Neurones Spiking LIF pour NATIVA-Learn
=======================================
Implémentation vectorisée du modèle Leaky Integrate-and-Fire (LIF).

Architecture:
- LIFNeuron : dataclass pour les paramètres d'un neurone unique
- NeuronPopulation : gestion vectorisée de N neurones (NumPy)
  avec support pour kill/resilience testing

Modèle LIF:
    τ_m · dV/dt = -(V - V_rest) + R_m · I(t)
    si V >= V_th  → spike, V ← V_reset, refractory pendant t_ref

REFERENCE: Gerstner, W. & Kistler, W. (2002). Spiking Neuron Models.
           Cambridge University Press. Chapter 1.3 "Leaky Integrate-and-Fire"

REFERENCE: Dayan, P. & Abbott, L.F. (2001). Theoretical Neuroscience.
           MIT Press. Eq. 1.1-1.5

Conventions de documentation pour audit LLM:
    # CONTRACT: pré/post-conditions d'une méthode
    # INVARIANT: propriété mathématique toujours vraie
    # REFERENCE: citation papier/équation source
    # VERIFY: assertion qu'un auditeur LLM doit vérifier
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ==============================================================================
# Paramètres d'un Neurone LIF
# ==============================================================================

@dataclass
class LIFNeuron:
    """
    Paramètres biologiques d'un neurone Leaky Integrate-and-Fire.

    Le LIF est le modèle de neurone spiking le plus simple et le mieux
    compris. Il capture l'essentiel : intégration, fuite, et seuil.

    REFERENCE: Gerstner (2002) Eq. 1.5 — Modèle LIF standard

    Attributes:
        V_rest: Potentiel de repos (mV). Le neurone revient ici sans stimulation.
        V_th: Seuil de déclenchement (mV). Un spike est émis quand V >= V_th.
        V_reset: Potentiel après un spike (mV). Toujours < V_th.
        tau_m: Constante de temps membranaire (ms). Contrôle la vitesse de fuite.
                Plus tau_m est grand, plus le neurone garde la mémoire de ses entrées.
        R_m: Résistance membranaire (MΩ). Amplifie le courant d'entrée.
        t_ref: Période réfractaire (ms). Après un spike, le neurone est "sourd"
               pendant t_ref millisecondes. Empêche les bursts trop rapides.

    INVARIANT: V_reset < V_th (sinon le neurone fire en boucle infinie)
    INVARIANT: tau_m > 0 (sinon la dynamique est indéfinie)
    INVARIANT: t_ref >= 0 (pas de période réfractaire négative)
    """
    V_rest: float = -65.0      # mV — potentiel de repos typique
    V_th: float = -50.0        # mV — seuil de déclenchement typique
    V_reset: float = -70.0     # mV — hyperpolarisation post-spike
    tau_m: float = 20.0        # ms — constante de temps membranaire (20ms = cortex)
    R_m: float = 10.0          # MΩ — résistance membranaire
    t_ref: float = 2.0         # ms — période réfractaire absolue

    def __post_init__(self):
        # VERIFY: Les invariants structurels sont respectés
        assert self.V_reset < self.V_th, \
            f"CONTRACT VIOLATION: V_reset ({self.V_reset}) doit être < V_th ({self.V_th})"
        assert self.tau_m > 0, \
            f"CONTRACT VIOLATION: tau_m ({self.tau_m}) doit être > 0"
        assert self.t_ref >= 0, \
            f"CONTRACT VIOLATION: t_ref ({self.t_ref}) doit être >= 0"


# ==============================================================================
# Population de Neurones (Vectorisé)
# ==============================================================================

class NeuronPopulation:
    """
    Population de N neurones LIF, entièrement vectorisée avec NumPy.

    C'est le building block de NATIVA-Learn. Une population est un ensemble
    de neurones qui partagent les mêmes paramètres mais ont des états
    indépendants (potentiel V, temps réfractaire, vivant/mort).

    REFERENCE: Brette, R. & Gerstner, W. (2005).
               "Adaptive Exponential Integrate-and-Fire Model"
               — Ici on utilise le LIF standard comme cas simplifié.

    Caractéristiques NATIVA:
    - kill(indices) : désactiver des neurones pour tester la résilience
    - alive_mask : masque booléen des neurones actifs
    - Toutes les opérations sont vectorisées O(N) via NumPy

    INVARIANT: À tout instant, V[i] <= V_th pour tout neurone i
               (car un spike remet immédiatement V à V_reset)
    INVARIANT: alive_mask[i] == False → V[i] == V_rest (neurone mort = au repos)
    INVARIANT: refractory_remaining[i] >= 0 toujours
    """

    def __init__(
        self,
        n_neurons: int,
        params: Optional[LIFNeuron] = None,
        seed: Optional[int] = None
    ):
        """
        Initialise une population de n_neurons neurones LIF.

        CONTRACT: n_neurons > 0
        CONTRACT: params est un LIFNeuron valide (invariants vérifiés)

        Args:
            n_neurons: Nombre de neurones dans la population.
            params: Paramètres LIF partagés. Si None, utilise les défauts.
            seed: Graine aléatoire pour reproductibilité. Si None, non fixée.
        """
        assert n_neurons > 0, f"CONTRACT VIOLATION: n_neurons ({n_neurons}) doit être > 0"

        self.n_neurons = n_neurons
        self.params = params or LIFNeuron()

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        # --- État interne (arrays NumPy pour vectorisation) ---

        # Potentiels membranaires : initialisés au repos
        # INVARIANT: V[i] <= V_th pour tout i
        self.V = np.full(n_neurons, self.params.V_rest, dtype=np.float64)

        # Compteur réfractaire restant (ms). 0 = le neurone peut être activé.
        # INVARIANT: refractory_remaining[i] >= 0 pour tout i
        self.refractory_remaining = np.zeros(n_neurons, dtype=np.float64)

        # Masque des neurones vivants. True = actif, False = détruit.
        # INVARIANT: alive_mask[i] == False → V[i] == V_rest
        self.alive_mask = np.ones(n_neurons, dtype=bool)

        # --- Compteurs et historique ---
        self.spike_counts = np.zeros(n_neurons, dtype=np.int64)
        self.total_steps = 0
        self.last_spike_time = np.full(n_neurons, -np.inf, dtype=np.float64)

    # ------------------------------------------------------------------
    # Dynamique LIF
    # ------------------------------------------------------------------

    def step(
        self,
        input_current: np.ndarray,
        dt: float = 1.0,
        current_time: float = 0.0
    ) -> np.ndarray:
        """
        Avance la simulation d'un pas de temps dt.

        Algorithme (pour chaque neurone vivant et non-réfractaire) :
            1. Calculer dV/dt = -(V - V_rest) / tau_m + R_m * I / tau_m
            2. Mettre à jour V += dV/dt * dt  (Euler explicite)
            3. Si V >= V_th → spike, V ← V_reset, refractory ← t_ref

        CONTRACT: input_current.shape == (n_neurons,)
        CONTRACT: dt > 0
        CONTRACT: Retourne un array booléen de shape (n_neurons,)

        REFERENCE: Gerstner (2002) Eq. 1.5 — Discrétisation Euler du LIF
                   τ_m · dV/dt = -(V - V_rest) + R_m · I(t)
                   → V(t+dt) = V(t) + dt/τ_m · [-(V(t) - V_rest) + R_m · I(t)]

        Args:
            input_current: Courant d'entrée I(t) pour chaque neurone. Shape (N,).
                          Unité : nA. Peut être négatif (inhibition).
            dt: Pas de temps en ms. Typiquement 0.1 à 1.0 ms.
            current_time: Temps absolu en ms (pour recorder les spike times).

        Returns:
            spikes: Array booléen (N,). True si le neurone i a émis un spike.
        """
        assert input_current.shape == (self.n_neurons,), \
            f"CONTRACT VIOLATION: input_current.shape={input_current.shape}, attendu ({self.n_neurons},)"
        assert dt > 0, f"CONTRACT VIOLATION: dt={dt} doit être > 0"

        p = self.params

        # Masque des neurones qui peuvent être mis à jour :
        # - vivants ET pas en période réfractaire
        active = self.alive_mask & (self.refractory_remaining <= 0)

        # --- Étape 1 : Calculer dV/dt (vectorisé) ---
        # REFERENCE: τ_m · dV/dt = -(V - V_rest) + R_m · I(t)
        # → dV/dt = [-(V - V_rest) + R_m · I] / τ_m
        dV = np.zeros(self.n_neurons, dtype=np.float64)
        dV[active] = (
            -(self.V[active] - p.V_rest) + p.R_m * input_current[active]
        ) / p.tau_m

        # --- Étape 2 : Mise à jour Euler ---
        # REFERENCE: V(t+dt) ≈ V(t) + dV/dt · dt
        self.V += dV * dt

        # --- Étape 3 : Détection des spikes ---
        # Un spike se produit quand V >= V_th ET le neurone est actif
        spikes = (self.V >= p.V_th) & self.alive_mask
        # VERIFY: Aucun neurone mort ne devrait produire de spike
        assert not np.any(spikes & ~self.alive_mask), \
            "INVARIANT VIOLATION: Un neurone mort a émis un spike"

        # --- Étape 4 : Reset post-spike ---
        # Les neurones qui ont spiké sont remis à V_reset
        self.V[spikes] = p.V_reset
        # Démarrer la période réfractaire
        self.refractory_remaining[spikes] = p.t_ref
        # Enregistrer le temps du spike
        self.last_spike_time[spikes] = current_time
        # Compteur
        self.spike_counts[spikes] += 1

        # --- Étape 5 : Décrémenter les périodes réfractaires ---
        self.refractory_remaining = np.maximum(0, self.refractory_remaining - dt)
        # INVARIANT: refractory_remaining >= 0 toujours
        assert np.all(self.refractory_remaining >= 0), \
            "INVARIANT VIOLATION: refractory_remaining négatif détecté"

        # --- Étape 6 : Forcer les neurones morts au repos ---
        self.V[~self.alive_mask] = p.V_rest

        # VERIFY: Après le reset, aucun V ne devrait dépasser V_th
        # (sauf transitoirement dans le même pas — le reset l'a corrigé)

        self.total_steps += 1
        return spikes

    # ------------------------------------------------------------------
    # Résilience : Kill & Resurrect
    # ------------------------------------------------------------------

    def kill(self, indices: np.ndarray) -> int:
        """
        Détruit des neurones (simule une panne matérielle ou dégradation).

        Le neurone tué :
        - Ne répond plus à aucun courant d'entrée
        - N'émet plus de spikes
        - Son potentiel est forcé à V_rest

        CONTRACT: indices contient des indices valides dans [0, n_neurons)
        CONTRACT: Retourne le nombre de neurones effectivement tués (pas déjà morts)

        INVARIANT: Après kill, alive_mask[indices] == False
        INVARIANT: Après kill, V[indices] == V_rest

        Args:
            indices: Indices des neurones à tuer. Array d'entiers.

        Returns:
            Nombre de neurones nouvellement tués.
        """
        indices = np.asarray(indices, dtype=int)
        assert np.all((indices >= 0) & (indices < self.n_neurons)), \
            f"CONTRACT VIOLATION: indices hors bornes [0, {self.n_neurons})"

        # Compter ceux qui étaient vivants
        newly_killed = np.sum(self.alive_mask[indices])

        # Tuer
        self.alive_mask[indices] = False
        self.V[indices] = self.params.V_rest
        self.refractory_remaining[indices] = 0.0

        return int(newly_killed)

    def kill_fraction(self, fraction: float) -> np.ndarray:
        """
        Détruit une fraction aléatoire des neurones VIVANTS.

        CONTRACT: 0 <= fraction <= 1
        CONTRACT: Retourne les indices des neurones tués

        Args:
            fraction: Proportion de neurones vivants à tuer (0.0 à 1.0).

        Returns:
            Indices des neurones tués.
        """
        assert 0.0 <= fraction <= 1.0, \
            f"CONTRACT VIOLATION: fraction={fraction} doit être dans [0, 1]"

        alive_indices = np.where(self.alive_mask)[0]
        n_to_kill = int(len(alive_indices) * fraction)

        if n_to_kill == 0:
            return np.array([], dtype=int)

        victims = self._rng.choice(alive_indices, size=n_to_kill, replace=False)
        self.kill(victims)
        return victims

    def resurrect(self, indices: np.ndarray):
        """
        Ressuscite des neurones (pour les tests, pas en production).

        CONTRACT: indices contient des indices valides dans [0, n_neurons)
        """
        indices = np.asarray(indices, dtype=int)
        self.alive_mask[indices] = True
        self.V[indices] = self.params.V_rest
        self.refractory_remaining[indices] = 0.0

    # ------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------

    def alive_ratio(self) -> float:
        """
        Fraction de neurones encore vivants.

        INVARIANT: 0 <= alive_ratio <= 1
        INVARIANT: alive_ratio == sum(alive_mask) / n_neurons

        Returns:
            Ratio ∈ [0.0, 1.0]
        """
        return float(np.sum(self.alive_mask) / self.n_neurons)

    def n_alive(self) -> int:
        """Nombre de neurones vivants."""
        return int(np.sum(self.alive_mask))

    def firing_rates(self, dt: float = 1.0) -> np.ndarray:
        """
        Taux de décharge moyen de chaque neurone (Hz).

        CONTRACT: dt > 0
        CONTRACT: total_steps > 0 (sinon division par zéro)

        REFERENCE: firing_rate_i = spike_count_i / (total_steps * dt / 1000)
                   (dt en ms, on divise par 1000 pour avoir des secondes)

        Returns:
            Array (N,) des taux de décharge en Hz.
        """
        if self.total_steps == 0:
            return np.zeros(self.n_neurons)
        total_time_s = self.total_steps * dt / 1000.0  # ms → s
        return self.spike_counts / total_time_s

    def get_diagnostics(self) -> Dict:
        """
        Retourne un snapshot complet de l'état de la population.

        Conçu pour être loggé ou affiché dans un dashboard.
        """
        return {
            'n_neurons': self.n_neurons,
            'n_alive': self.n_alive(),
            'alive_ratio': self.alive_ratio(),
            'total_spikes': int(np.sum(self.spike_counts)),
            'total_steps': self.total_steps,
            'mean_V': float(np.mean(self.V[self.alive_mask])) if self.n_alive() > 0 else 0.0,
            'std_V': float(np.std(self.V[self.alive_mask])) if self.n_alive() > 0 else 0.0,
            'max_spike_count': int(np.max(self.spike_counts)),
        }

    def reset(self):
        """Remet tous les neurones à leur état initial (sans changer alive_mask)."""
        self.V[:] = self.params.V_rest
        self.refractory_remaining[:] = 0.0
        self.spike_counts[:] = 0
        self.total_steps = 0
        self.last_spike_time[:] = -np.inf


# ==============================================================================
# Tests et Démonstration
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODULE NEURON (LIF)")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Test 1 : Création et paramètres
    # ---------------------------------------------------------------
    print("\n--- Test 1: Création d'une population ---")
    pop = NeuronPopulation(n_neurons=100, seed=42)
    print(f"Population créée: {pop.n_neurons} neurones")
    print(f"Paramètres: V_rest={pop.params.V_rest}, V_th={pop.params.V_th}, "
          f"tau_m={pop.params.tau_m}ms")
    print(f"Alive ratio: {pop.alive_ratio():.2f}")
    assert pop.alive_ratio() == 1.0, "VERIFY FAILED: Tous les neurones devraient être vivants"
    print("✅ Création OK")

    # ---------------------------------------------------------------
    # Test 2 : Decay sans entrée (V doit converger vers V_rest)
    # ---------------------------------------------------------------
    print("\n--- Test 2: Decay sans entrée ---")
    pop2 = NeuronPopulation(n_neurons=10, seed=42)
    # Initialiser à un potentiel au-dessus du repos mais sous le seuil
    pop2.V[:] = -55.0  # Entre V_rest (-65) et V_th (-50)

    dt = 0.5  # ms
    for step in range(200):  # 100 ms de simulation
        spikes = pop2.step(np.zeros(10), dt=dt, current_time=step * dt)

    final_V = pop2.V[0]
    print(f"V initial: -55.0 mV, V final: {final_V:.4f} mV, V_rest: {pop2.params.V_rest}")
    # VERIFY: V doit converger vers V_rest
    assert abs(final_V - pop2.params.V_rest) < 0.1, \
        f"VERIFY FAILED: V ({final_V:.4f}) devrait converger vers V_rest ({pop2.params.V_rest})"
    print("✅ Decay OK — V converge vers V_rest")

    # ---------------------------------------------------------------
    # Test 3 : Spike avec courant fort
    # ---------------------------------------------------------------
    print("\n--- Test 3: Spike avec courant constant ---")
    pop3 = NeuronPopulation(n_neurons=1, seed=42)
    dt = 0.5
    total_spikes = 0
    for step in range(200):  # 100 ms
        spikes = pop3.step(np.array([3.0]), dt=dt, current_time=step * dt)  # Courant fort
        total_spikes += np.sum(spikes)

    print(f"Courant: 3.0 nA pendant 100ms → {total_spikes} spikes")
    # VERIFY: Un courant fort doit provoquer au moins 1 spike
    assert total_spikes > 0, "VERIFY FAILED: Le neurone devrait avoir spiké avec un courant fort"
    print("✅ Spike OK — Le neurone répond au courant")

    # ---------------------------------------------------------------
    # Test 4 : Résilience — kill
    # ---------------------------------------------------------------
    print("\n--- Test 4: Résilience — kill ---")
    pop4 = NeuronPopulation(n_neurons=100, seed=42)
    
    # Tuer 50% des neurones
    killed = pop4.kill_fraction(0.5)
    print(f"Neurones tués: {len(killed)}")
    print(f"Alive ratio: {pop4.alive_ratio():.2f}")
    assert abs(pop4.alive_ratio() - 0.5) < 0.02, \
        f"VERIFY FAILED: alive_ratio ({pop4.alive_ratio()}) devrait être ~0.5"

    # VERIFY: Les neurones morts ne peuvent pas spiker
    dt = 0.5
    for step in range(100):
        spikes = pop4.step(np.full(100, 3.0), dt=dt, current_time=step * dt)
        dead_spikes = np.sum(spikes & ~pop4.alive_mask)
        assert dead_spikes == 0, "VERIFY FAILED: Un neurone mort a émis un spike"

    alive_spike_count = np.sum(pop4.spike_counts[pop4.alive_mask])
    dead_spike_count = np.sum(pop4.spike_counts[~pop4.alive_mask])
    print(f"Spikes des vivants: {alive_spike_count}, Spikes des morts: {dead_spike_count}")
    assert dead_spike_count == 0, "VERIFY FAILED: Les neurones morts ont des spikes dans le compteur"
    print("✅ Kill OK — Les neurones morts sont silencieux")

    # ---------------------------------------------------------------
    # Test 5 : Période réfractaire
    # ---------------------------------------------------------------
    print("\n--- Test 5: Période réfractaire ---")
    pop5 = NeuronPopulation(
        n_neurons=1,
        params=LIFNeuron(t_ref=5.0),  # 5ms de réfractaire
        seed=42
    )
    dt = 0.5  # pas de 0.5ms
    spike_times = []
    for step in range(200):
        t = step * dt
        spikes = pop5.step(np.array([3.0]), dt=dt, current_time=t)
        if spikes[0]:
            spike_times.append(t)

    if len(spike_times) >= 2:
        # VERIFY: L'intervalle inter-spike doit être >= t_ref
        intervals = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
        min_interval = min(intervals)
        print(f"Spike times (premiers 5): {spike_times[:5]}")
        print(f"Intervalle inter-spike minimum: {min_interval:.1f} ms (t_ref={pop5.params.t_ref} ms)")
        assert min_interval >= pop5.params.t_ref - dt, \
            f"VERIFY FAILED: ISI ({min_interval:.1f}ms) < t_ref ({pop5.params.t_ref}ms)"
        print("✅ Réfractaire OK — ISI >= t_ref")
    else:
        print("⚠️ Pas assez de spikes pour tester le réfractaire")

    # ---------------------------------------------------------------
    # Test 6 : Diagnostics
    # ---------------------------------------------------------------
    print("\n--- Test 6: Diagnostics ---")
    diag = pop4.get_diagnostics()
    print(f"Diagnostics: {diag}")
    assert diag['n_neurons'] == 100
    assert diag['n_alive'] == 50
    print("✅ Diagnostics OK")

    # ---------------------------------------------------------------
    # Résumé
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RÉSULTAT: ✅ Tous les tests neuron.py passent")
    print("=" * 60)
    print("\nPropriétés vérifiées:")
    print("  ✓ V converge vers V_rest sans entrée (decay)")
    print("  ✓ Le neurone spike avec un courant fort")
    print("  ✓ Les neurones morts ne spikent jamais")
    print("  ✓ La période réfractaire est respectée")
    print("  ✓ alive_ratio est correct après kill")


tajweed_params = LIFNeuron(
    tau_m=100.0,
    V_th=45.0,     # On remonte de 10 à 15
    R_m=50.0,
    t_ref=3.0
)