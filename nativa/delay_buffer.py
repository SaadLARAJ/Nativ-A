"""
Buffer de Délais pour Secteur 0
================================
Implémente un buffer circulaire avec interpolation pour gérer
les délais de communication réalistes (50-200ms).

Basé sur les recommandations de l'expert:
- Buffer circulaire de taille fixe
- Interpolation linéaire pour délais non-multiples de dt
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class DelayBuffer:
    """
    Buffer circulaire pour stocker l'historique d'états.
    
    Permet de récupérer l'état d'un nœud à un temps passé (t - τ).
    
    Attributes:
        max_delay: Délai maximum gérable (en secondes)
        dt: Pas de temps de la simulation
        buffer_size: Nombre d'états stockés
    """
    max_delay: float  # Délai max en secondes (ex: 0.2 pour 200ms)
    dt: float  # Pas de temps (ex: 0.01 pour 10ms)
    
    # Calculé automatiquement
    buffer_size: int = field(init=False)
    _buffer: deque = field(init=False, repr=False)
    _times: deque = field(init=False, repr=False)
    
    def __post_init__(self):
        # Taille = délai_max / dt, arrondi au supérieur + 1 pour interpolation
        self.buffer_size = int(np.ceil(self.max_delay / self.dt)) + 2
        self._buffer = deque(maxlen=self.buffer_size)
        self._times = deque(maxlen=self.buffer_size)
    
    def push(self, state: np.ndarray, time: float):
        """
        Ajoute un nouvel état au buffer.
        
        Args:
            state: État à stocker (copié pour éviter les références)
            time: Temps associé à cet état
        """
        self._buffer.append(state.copy())
        self._times.append(time)
    
    def get_delayed(self, delay: float, current_time: float) -> Optional[np.ndarray]:
        """
        Récupère l'état à (current_time - delay) avec interpolation.
        
        Args:
            delay: Délai τ en secondes
            current_time: Temps actuel
        
        Returns:
            État interpolé à t - τ, ou None si pas assez d'historique
        """
        if len(self._buffer) < 2:
            return None
        
        target_time = current_time - delay
        
        # Chercher les deux points encadrant target_time
        times = list(self._times)
        states = list(self._buffer)
        
        # Si target_time est avant le premier état stocké
        if target_time < times[0]:
            return states[0]  # Retourner le plus ancien disponible
        
        # Si target_time est après le dernier état (ne devrait pas arriver)
        if target_time >= times[-1]:
            return states[-1]
        
        # Trouver l'intervalle [t_i, t_{i+1}] contenant target_time
        for i in range(len(times) - 1):
            if times[i] <= target_time < times[i + 1]:
                # Interpolation linéaire
                t0, t1 = times[i], times[i + 1]
                s0, s1 = states[i], states[i + 1]
                
                alpha = (target_time - t0) / (t1 - t0) if t1 != t0 else 0
                return s0 + alpha * (s1 - s0)
        
        return states[-1]
    
    def is_ready(self, delay: float) -> bool:
        """Vérifie si le buffer a assez d'historique pour ce délai."""
        if len(self._times) < 2:
            return False
        return (self._times[-1] - self._times[0]) >= delay
    
    def clear(self):
        """Vide le buffer."""
        self._buffer.clear()
        self._times.clear()


class DelayNetwork:
    """
    Gestionnaire de délais pour un réseau de N nœuds.
    
    Chaque nœud a son propre buffer et une matrice de délais τ_ij
    définit le délai de communication entre chaque paire.
    """
    
    def __init__(
        self, 
        n_nodes: int, 
        state_dim: int,
        dt: float = 0.01,
        delay_min: float = 0.05,  # 50ms
        delay_max: float = 0.2,   # 200ms
        delay_matrix: Optional[np.ndarray] = None
    ):
        """
        Args:
            n_nodes: Nombre de nœuds
            state_dim: Dimension de l'état de chaque nœud
            dt: Pas de temps
            delay_min: Délai minimum (pour génération aléatoire)
            delay_max: Délai maximum
            delay_matrix: Matrice NxN des délais. Si None, générée aléatoirement.
        """
        self.n_nodes = n_nodes
        self.state_dim = state_dim
        self.dt = dt
        
        # Créer la matrice de délais
        if delay_matrix is not None:
            self.delays = delay_matrix
        else:
            # Générer des délais aléatoires symétriques
            self.delays = np.random.uniform(delay_min, delay_max, (n_nodes, n_nodes))
            # Rendre symétrique (délai i→j = délai j→i)
            self.delays = (self.delays + self.delays.T) / 2
            # Pas de délai avec soi-même
            np.fill_diagonal(self.delays, 0)
        
        # Buffer par nœud
        max_delay = np.max(self.delays)
        self.buffers: List[DelayBuffer] = [
            DelayBuffer(max_delay=max_delay, dt=dt) 
            for _ in range(n_nodes)
        ]
    
    def push_states(self, states: np.ndarray, time: float):
        """
        Met à jour tous les buffers avec les nouveaux états.
        
        Args:
            states: Array (n_nodes, state_dim) des états
            time: Temps actuel
        """
        for i, buffer in enumerate(self.buffers):
            buffer.push(states[i], time)
    
    def get_delayed_state(self, from_node: int, to_node: int, current_time: float) -> Optional[np.ndarray]:
        """
        Récupère l'état de from_node tel que vu par to_node (avec délai).
        
        Args:
            from_node: Nœud source
            to_node: Nœud destination
            current_time: Temps actuel
        
        Returns:
            État de from_node à (t - τ_{from,to})
        """
        delay = self.delays[from_node, to_node]
        return self.buffers[from_node].get_delayed(delay, current_time)
    
    def get_all_delayed_for_node(self, node: int, current_time: float) -> Dict[int, np.ndarray]:
        """
        Récupère les états retardés de tous les autres nœuds pour un nœud donné.
        
        Args:
            node: Nœud qui observe
            current_time: Temps actuel
        
        Returns:
            Dict {autre_nœud: état_retardé}
        """
        delayed_states = {}
        for other in range(self.n_nodes):
            if other != node:
                state = self.get_delayed_state(other, node, current_time)
                if state is not None:
                    delayed_states[other] = state
        return delayed_states
    
    def is_warmed_up(self) -> bool:
        """Vérifie si tous les buffers ont assez d'historique."""
        max_delay = np.max(self.delays)
        return all(buf.is_ready(max_delay) for buf in self.buffers)
    
    def get_delay_stats(self) -> Dict:
        """Retourne des statistiques sur les délais."""
        non_zero = self.delays[self.delays > 0]
        return {
            'min_delay_ms': np.min(non_zero) * 1000 if len(non_zero) > 0 else 0,
            'max_delay_ms': np.max(non_zero) * 1000 if len(non_zero) > 0 else 0,
            'mean_delay_ms': np.mean(non_zero) * 1000 if len(non_zero) > 0 else 0,
        }


# ==============================================================================
# Tests et Démonstration
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU BUFFER DE DÉLAIS")
    print("=" * 60)
    
    # Test 1: Buffer simple
    print("\n--- Test 1: Buffer simple ---")
    buffer = DelayBuffer(max_delay=0.2, dt=0.01)
    print(f"Buffer créé: taille={buffer.buffer_size}, max_delay=200ms, dt=10ms")
    
    # Remplir le buffer
    for step in range(30):
        t = step * 0.01
        state = np.array([step * 1.0])  # État = numéro du pas
        buffer.push(state, t)
    
    # Tester l'interpolation
    current_time = 0.29
    delays_to_test = [0.05, 0.1, 0.15, 0.175, 0.2]
    
    print(f"\nTemps actuel: {current_time*1000:.0f}ms")
    for delay in delays_to_test:
        state = buffer.get_delayed(delay, current_time)
        expected = (current_time - delay) / 0.01
        print(f"  Délai {delay*1000:.0f}ms → État={state[0]:.2f} (attendu~{expected:.1f})")
    
    # Test 2: Réseau de délais
    print("\n--- Test 2: Réseau de délais ---")
    network = DelayNetwork(n_nodes=5, state_dim=2, dt=0.01)
    print(f"Délais générés: {network.get_delay_stats()}")
    
    # Simuler quelques pas
    states = np.random.randn(5, 2)
    for step in range(50):
        t = step * 0.01
        network.push_states(states, t)
        states = states + np.random.randn(5, 2) * 0.01  # Évolution aléatoire
    
    print(f"Buffer prêt: {network.is_warmed_up()}")
    
    # Récupérer les états retardés
    delayed = network.get_all_delayed_for_node(0, current_time=0.49)
    print(f"États retardés pour nœud 0: {len(delayed)} voisins")
    
    print("\n✅ Module delay_buffer.py opérationnel")
