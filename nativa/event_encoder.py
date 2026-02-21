"""
Encodeurs Événementiels pour NATIVA-Learn
==========================================
Convertit des données brutes (valeurs, images, séries temporelles) en
trains de spikes compatibles avec les neurones LIF.

Architecture:
    - RateEncoder : valeur → probabilité de spike (rate coding de Poisson)
    - TemporalEncoder : valeur → latence du spike (temporal coding)
    - DeltaEncoder : différence entre frames → événements ON/OFF (simule DVS)

Pourquoi encoder en spikes ?
    Les SNN (Spiking Neural Networks) traitent de l'information temporelle.
    Pour pouvoir apprendre avec STDP, les données doivent être converties
    en séquences de spikes (0 ou 1 par pas de temps).

REFERENCE: Auge, D., Hille, J., Mueller, E. & Knoll, A. (2021).
           "A Survey of Encoding Techniques for Signal Processing in SNNs"
           Neural Processing Letters, 53, 4693-4710.

REFERENCE: Gallego, G. et al. (2020).
           "Event-based Vision: A Survey"
           IEEE TPAMI. — Pour le DeltaEncoder (DVS)

Conventions de documentation pour audit LLM:
    # CONTRACT: pré/post-conditions
    # INVARIANT: propriété mathématique toujours vraie
    # REFERENCE: citation papier/équation source
    # VERIFY: assertion pour l'auditeur
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Generator, Tuple, List


# ==============================================================================
# Rate Encoder (Codage en fréquence)
# ==============================================================================

class RateEncoder:
    """
    Encode des valeurs réelles en trains de spikes via un processus de Poisson.

    Principe : une valeur élevée → haute probabilité de spike à chaque pas.

    Pour une valeur v ∈ [v_min, v_max] :
        p_spike = (v - v_min) / (v_max - v_min)  (normalisé dans [0, 1])
        spike ~ Bernoulli(p_spike)                 (tirage aléatoire)

    Sur N pas de temps, le taux de spike converge vers p_spike (loi des grands nombres).

    REFERENCE: Auge et al. (2021) Section 3.1 "Rate Coding"
    REFERENCE: Adrian, E.D. (1926) — Découverte du rate coding biologique

    INVARIANT: 0 <= p_spike <= 1 pour tout v ∈ [v_min, v_max]
    INVARIANT: valeur haute → plus de spikes (monotone)
    """

    def __init__(
        self,
        v_min: float = 0.0,
        v_max: float = 1.0,
        n_steps: int = 20,
        seed: Optional[int] = None
    ):
        """
        CONTRACT: v_max > v_min
        CONTRACT: n_steps > 0

        Args:
            v_min: Valeur minimale attendue dans les données.
            v_max: Valeur maximale attendue dans les données.
            n_steps: Nombre de pas de temps pour encoder une valeur.
                     Plus de pas = meilleure résolution mais plus lent.
            seed: Graine aléatoire pour reproductibilité.
        """
        assert v_max > v_min, f"CONTRACT VIOLATION: v_max ({v_max}) doit être > v_min ({v_min})"
        assert n_steps > 0, f"CONTRACT VIOLATION: n_steps ({n_steps}) doit être > 0"

        self.v_min = v_min
        self.v_max = v_max
        self.n_steps = n_steps
        self._rng = np.random.default_rng(seed)

    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        Encode un vecteur de valeurs en une séquence de spikes.

        CONTRACT: values.shape == (n_features,)
        CONTRACT: Retourne un array (n_steps, n_features) de 0/1

        REFERENCE: p_spike = clip((v - v_min) / (v_max - v_min), 0, 1)
                   spike[t, f] ~ Bernoulli(p_spike[f])

        Args:
            values: Vecteur de valeurs à encoder. Shape (n_features,).

        Returns:
            Spike train de shape (n_steps, n_features). Chaque élément est 0 ou 1.
        """
        values = np.asarray(values, dtype=np.float64)
        n_features = len(values)

        # Normaliser dans [0, 1]
        # REFERENCE: p = (v - v_min) / (v_max - v_min)
        p_spike = np.clip(
            (values - self.v_min) / (self.v_max - self.v_min),
            0.0, 1.0
        )
        # INVARIANT: 0 <= p_spike <= 1

        # Générer les spikes (Bernoulli)
        random_matrix = self._rng.random((self.n_steps, n_features))
        spikes = (random_matrix < p_spike).astype(np.float64)

        return spikes

    def encode_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Encode un batch de N vecteurs.

        CONTRACT: batch.shape == (N, n_features)
        CONTRACT: Retourne (N, n_steps, n_features)

        Args:
            batch: Batch de valeurs. Shape (N, n_features).

        Returns:
            Spike trains. Shape (N, n_steps, n_features).
        """
        N, n_features = batch.shape
        result = np.zeros((N, self.n_steps, n_features), dtype=np.float64)
        for i in range(N):
            result[i] = self.encode(batch[i])
        return result


# ==============================================================================
# Temporal Encoder (Codage en latence)
# ==============================================================================

class TemporalEncoder:
    """
    Encode des valeurs en latence de spike (time-to-first-spike).

    Principe : une valeur élevée → spike TÔT, une valeur basse → spike TARD.
    Un seul spike par neurone par présentation.

    Pour une valeur v ∈ [v_min, v_max] :
        latence = n_steps * (1 - (v - v_min) / (v_max - v_min))
        spike à t = round(latence)

    REFERENCE: Auge et al. (2021) Section 3.2 "Temporal Coding"
    REFERENCE: Thorpe, S. & Gautrais, J. (1998).
               "Rank order coding" — Les premiers spikes portent le plus d'info

    INVARIANT: Chaque neurone spike exactement une fois
    INVARIANT: Valeur haute → latence basse (monotone inversée)
    """

    def __init__(
        self,
        v_min: float = 0.0,
        v_max: float = 1.0,
        n_steps: int = 20
    ):
        """
        CONTRACT: v_max > v_min
        CONTRACT: n_steps > 0
        """
        assert v_max > v_min, f"CONTRACT VIOLATION: v_max ({v_max}) doit être > v_min ({v_min})"
        assert n_steps > 0, f"CONTRACT VIOLATION: n_steps ({n_steps}) doit être > 0"

        self.v_min = v_min
        self.v_max = v_max
        self.n_steps = n_steps

    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        Encode un vecteur en time-to-first-spike.

        CONTRACT: values.shape == (n_features,)
        CONTRACT: Retourne (n_steps, n_features) avec exactement un 1 par colonne

        REFERENCE: latence = n_steps · (1 - normalized_value)
                   → valeur haute = spike tôt (latence basse)

        Args:
            values: Valeurs à encoder. Shape (n_features,).

        Returns:
            Spike train (n_steps, n_features). Un seul 1 par colonne.
        """
        values = np.asarray(values, dtype=np.float64)
        n_features = len(values)

        # Normaliser
        normalized = np.clip(
            (values - self.v_min) / (self.v_max - self.v_min),
            0.0, 1.0
        )

        # Calculer les latences (inversées : haute valeur → basse latence)
        # REFERENCE: latence = n_steps * (1 - normalized)
        latencies = np.round((self.n_steps - 1) * (1.0 - normalized)).astype(int)
        latencies = np.clip(latencies, 0, self.n_steps - 1)

        # Construire le spike train
        spikes = np.zeros((self.n_steps, n_features), dtype=np.float64)
        for f in range(n_features):
            spikes[latencies[f], f] = 1.0

        return spikes


# ==============================================================================
# Delta Encoder (Simule une caméra DVS)
# ==============================================================================

class DeltaEncoder:
    """
    Encode les différences entre frames successives en événements ON/OFF.

    Simule le fonctionnement d'une caméra événementielle (DVS — Dynamic
    Vision Sensor). Seuls les pixels qui CHANGENT génèrent un événement.

    Pour chaque pixel :
        diff = frame_current - frame_previous
        si diff > threshold  → événement ON  (spike +1)
        si diff < -threshold → événement OFF (spike -1)
        sinon                → pas d'événement (0)

    REFERENCE: Gallego et al. (2020) "Event-based Vision: A Survey"
               IEEE TPAMI. Section 2 "Event Camera Model"

    REFERENCE: Lichtsteiner, P., Posch, C. & Delbruck, T. (2008).
               "A 128×128 120 dB 15 μs Latency Asynchronous Temporal
               Contrast Vision Sensor" — Le DVS original

    INVARIANT: encode_frame retourne des valeurs dans {-1, 0, +1}
    INVARIANT: Pas d'événement si la frame ne change pas
    """

    def __init__(
        self,
        threshold: float = 0.1,
        shape: Optional[Tuple[int, ...]] = None
    ):
        """
        CONTRACT: threshold > 0

        Args:
            threshold: Seuil de changement pour déclencher un événement.
                      Plus bas = plus sensible. Typiquement 0.05-0.3.
            shape: Shape optionnelle pour valider les frames d'entrée.
        """
        assert threshold > 0, f"CONTRACT VIOLATION: threshold ({threshold}) doit être > 0"

        self.threshold = threshold
        self.shape = shape
        self._previous_frame: Optional[np.ndarray] = None

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode les changements entre cette frame et la précédente.

        CONTRACT: frame est un array NumPy (accepte n'importe quelle shape)
        CONTRACT: Retourne un array de même shape avec valeurs dans {-1, 0, +1}

        REFERENCE: Gallego et al. (2020) Eq. 1
                   e_k = sign(L(x, t) - L(x, t - Δt)) si |ΔL| > C

        Args:
            frame: Frame actuelle (image ou vecteur de capteurs).

        Returns:
            Événements : +1 (ON), -1 (OFF), ou 0 (pas de changement).
        """
        frame = np.asarray(frame, dtype=np.float64)

        if self._previous_frame is None:
            # Première frame : pas de référence, pas d'événement
            self._previous_frame = frame.copy()
            return np.zeros_like(frame)

        # Calculer la différence
        diff = frame - self._previous_frame

        # Appliquer le seuil
        events = np.zeros_like(frame)
        events[diff > self.threshold] = 1.0    # ON
        events[diff < -self.threshold] = -1.0  # OFF

        # INVARIANT: événements ∈ {-1, 0, +1}
        assert np.all(np.isin(events, [-1.0, 0.0, 1.0])), \
            "INVARIANT VIOLATION: événements hors {-1, 0, +1}"

        # Mettre à jour la frame précédente
        self._previous_frame = frame.copy()

        return events

    def encode_stream(self, frames: np.ndarray) -> np.ndarray:
        """
        Encode un flux de T frames.

        CONTRACT: frames.shape == (T, ...) où T est le nombre de frames
        CONTRACT: Retourne (T, ...) d'événements

        Args:
            frames: Séquence de frames. Shape (T, *spatial_dims).

        Returns:
            Séquence d'événements. Shape (T, *spatial_dims).
        """
        self.reset()
        T = frames.shape[0]
        events = np.zeros_like(frames)
        for t in range(T):
            events[t] = self.encode_frame(frames[t])
        return events

    def reset(self):
        """Réinitialise la frame de référence."""
        self._previous_frame = None

    def get_event_rate(self, events: np.ndarray) -> float:
        """
        Calcule le taux d'événements (fraction de pixels actifs).

        Métrique utile pour évaluer la "densité" de la scène.

        Returns:
            Ratio ∈ [0, 1] de pixels qui ont généré un événement.
        """
        return float(np.mean(np.abs(events) > 0))


# ==============================================================================
# Tests et Démonstration
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODULE EVENT_ENCODER")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Test 1 : Rate Encoder — propriétés de base
    # ---------------------------------------------------------------
    print("\n--- Test 1: Rate Encoder ---")
    enc = RateEncoder(v_min=0.0, v_max=1.0, n_steps=1000, seed=42)

    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    spikes = enc.encode(values)

    print(f"  Shape: {spikes.shape} (attendu: (1000, 5))")
    assert spikes.shape == (1000, 5), "VERIFY FAILED: Shape incorrecte"

    # Vérifier que le taux de spike est proportionnel à la valeur
    rates = np.mean(spikes, axis=0)
    print(f"  Valeurs:  {values}")
    print(f"  Taux:     {np.round(rates, 3)}")

    # VERIFY: Monotonie — valeur plus haute → taux plus haut
    for i in range(len(rates) - 1):
        assert rates[i] <= rates[i+1] + 0.05, \
            f"VERIFY FAILED: Rate coding non monotone: rate[{i}]={rates[i]:.3f} > rate[{i+1}]={rates[i+1]:.3f}"

    # VERIFY: Le taux est approximativement égal à la valeur (avec tolérance)
    for i, (v, r) in enumerate(zip(values, rates)):
        assert abs(r - v) < 0.1, \
            f"VERIFY FAILED: Taux ({r:.3f}) trop loin de la valeur ({v:.3f})"

    print("✅ Rate Encoder OK — Taux proportionnel à la valeur")

    # ---------------------------------------------------------------
    # Test 2 : Temporal Encoder — time-to-first-spike
    # ---------------------------------------------------------------
    print("\n--- Test 2: Temporal Encoder ---")
    tenc = TemporalEncoder(v_min=0.0, v_max=1.0, n_steps=20)

    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    spikes = tenc.encode(values)

    print(f"  Shape: {spikes.shape} (attendu: (20, 5))")
    assert spikes.shape == (20, 5), "VERIFY FAILED: Shape incorrecte"

    # VERIFY: Exactement un spike par neurone
    spike_counts = np.sum(spikes, axis=0)
    assert np.all(spike_counts == 1), \
        f"VERIFY FAILED: Chaque neurone devrait avoir exactement 1 spike, got {spike_counts}"

    # Trouver les latences
    latencies = np.argmax(spikes, axis=0)
    print(f"  Valeurs:   {values}")
    print(f"  Latences:  {latencies}")

    # VERIFY: Valeur haute → latence basse (monotone inverse)
    for i in range(len(latencies) - 1):
        assert latencies[i] >= latencies[i+1], \
            f"VERIFY FAILED: Latence non décroissante: lat[{i}]={latencies[i]} < lat[{i+1}]={latencies[i+1]}"

    print("✅ Temporal Encoder OK — Valeur haute → spike tôt")

    # ---------------------------------------------------------------
    # Test 3 : Delta Encoder — simulation DVS
    # ---------------------------------------------------------------
    print("\n--- Test 3: Delta Encoder (DVS) ---")
    dvs = DeltaEncoder(threshold=0.1)

    # Créer 3 frames : identique, changement positif, changement négatif
    frame1 = np.array([0.5, 0.5, 0.5, 0.5])
    frame2 = np.array([0.5, 0.8, 0.5, 0.2])  # [0, +0.3, 0, -0.3]
    frame3 = np.array([0.5, 0.5, 0.5, 0.5])  # Retour à frame1

    events1 = dvs.encode_frame(frame1)  # Première frame → pas d'événement
    events2 = dvs.encode_frame(frame2)  # Diff avec frame1
    events3 = dvs.encode_frame(frame3)  # Diff avec frame2

    print(f"  Frame 1 → events: {events1} (première frame, pas de réf)")
    print(f"  Frame 2 → events: {events2} (attendu: [0, +1, 0, -1])")
    print(f"  Frame 3 → events: {events3} (attendu: [0, -1, 0, +1])")

    # VERIFY: Première frame = pas d'événement
    assert np.all(events1 == 0), "VERIFY FAILED: Première frame devrait être tout zéro"

    # VERIFY: Frame 2 détecte les changements
    assert events2[1] == 1.0, "VERIFY FAILED: Devrait détecter ON pour +0.3"
    assert events2[3] == -1.0, "VERIFY FAILED: Devrait détecter OFF pour -0.3"
    assert events2[0] == 0.0, "VERIFY FAILED: Pas de changement → pas d'événement"

    # VERIFY: événements ∈ {-1, 0, +1}
    assert np.all(np.isin(events2, [-1.0, 0.0, 1.0])), \
        "VERIFY FAILED: Événements hors {-1, 0, +1}"

    print("✅ Delta Encoder OK — DVS détecte ON/OFF correctement")

    # ---------------------------------------------------------------
    # Test 4 : Delta Encoder — flux (stream)
    # ---------------------------------------------------------------
    print("\n--- Test 4: Delta Encoder — stream ---")
    dvs2 = DeltaEncoder(threshold=0.05)

    # Créer un signal sinusoïdal (10 frames, 4 capteurs)
    t = np.linspace(0, 2 * np.pi, 20)
    frames = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2 * t),
        np.zeros_like(t)  # constant → pas d'événement
    ])

    events = dvs2.encode_stream(frames)
    print(f"  Frames shape: {frames.shape}")
    print(f"  Events shape: {events.shape}")

    # Taux d'événements par canal
    for ch in range(4):
        rate = np.mean(np.abs(events[:, ch]) > 0)
        print(f"  Canal {ch}: event rate = {rate:.2%}")

    # VERIFY: Le canal constant (3) ne devrait avoir presque aucun événement
    constant_rate = np.mean(np.abs(events[:, 3]) > 0)
    assert constant_rate == 0.0, \
        f"VERIFY FAILED: Canal constant devrait avoir 0 événements, got {constant_rate:.2%}"

    # VERIFY: Les canaux dynamiques devraient avoir des événements
    dynamic_rate = np.mean(np.abs(events[:, 0]) > 0)
    assert dynamic_rate > 0, "VERIFY FAILED: Le canal sinusoïdal devrait avoir des événements"

    print("✅ Stream OK — Canaux dynamiques actifs, canal constant silencieux")

    # ---------------------------------------------------------------
    # Test 5 : Round-trip Rate Encoder (encode → decode)
    # ---------------------------------------------------------------
    print("\n--- Test 5: Round-trip (encode → taux moyen ≈ valeur originale) ---")
    enc5 = RateEncoder(v_min=0.0, v_max=1.0, n_steps=5000, seed=42)

    original = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    spikes5 = enc5.encode(original)
    reconstructed = np.mean(spikes5, axis=0)

    print(f"  Original:      {original}")
    print(f"  Reconstruit:   {np.round(reconstructed, 3)}")

    max_error = np.max(np.abs(original - reconstructed))
    print(f"  Erreur max:    {max_error:.4f}")

    # VERIFY: L'erreur de reconstruction doit être petite avec 5000 pas
    assert max_error < 0.05, \
        f"VERIFY FAILED: Erreur de reconstruction ({max_error:.4f}) trop grande"

    print("✅ Round-trip OK — Reconstruction fidèle")

    # ---------------------------------------------------------------
    # Résumé
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RÉSULTAT: ✅ Tous les tests event_encoder.py passent")
    print("=" * 60)
    print("\nPropriétés vérifiées:")
    print("  ✓ Rate coding : taux proportionnel à la valeur (monotone)")
    print("  ✓ Rate coding : round-trip encode/decode fidèle (<5% erreur)")
    print("  ✓ Temporal coding : exactement un spike par neurone")
    print("  ✓ Temporal coding : valeur haute → spike tôt (monotone inverse)")
    print("  ✓ DVS : détecte ON (+1) et OFF (-1) correctement")
    print("  ✓ DVS : canal constant → zéro événement")
    print("  ✓ DVS : événements ∈ {-1, 0, +1}")
