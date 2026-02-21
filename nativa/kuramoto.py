"""
Kuramoto-Sakaguchi avec Délais pour Secteur 0
==============================================
Modèle de synchronisation réaliste avec:
- Délais de communication τ_ij
- Paramètre de frustration α (latence hardware)
- Intégration RK4
- Graphe de connectivité sparse
"""

import numpy as np
from typing import Optional, Callable, List, Dict, Tuple
from dataclasses import dataclass, field

try:
    from .delay_buffer import DelayNetwork
    from .integrators import rk4, rk4_with_monitoring, IntegrationDiagnostics
except ImportError:
    from delay_buffer import DelayNetwork
    from integrators import rk4, rk4_with_monitoring, IntegrationDiagnostics


@dataclass
class KuramotoNode:
    """
    Un oscillateur de Kuramoto avec paramètres individuels.
    
    Attributes:
        id: Identifiant du nœud
        omega: Fréquence naturelle
        alpha: Déphasage interne (frustration) - simule la latence hardware
        phase: Phase actuelle θ
    """
    id: int
    omega: float  # Fréquence naturelle
    alpha: float = 0.0  # Frustration (latence hardware)
    phase: float = 0.0  # Phase θ


class KuramotoSakaguchiNetwork:
    """
    Réseau de Kuramoto-Sakaguchi avec délais réalistes.
    
    Équation:
        dθ_i/dt = ω_i + (K/N) Σ_j A_ij sin(θ_j(t - τ_ij) - θ_i(t) + α_i)
    
    Où:
        - ω_i: fréquence naturelle du nœud i
        - K: force de couplage global
        - A_ij: matrice d'adjacence (1 si connecté, 0 sinon)
        - τ_ij: délai de communication entre i et j
        - α_i: déphasage interne (frustration) du nœud i
    """
    
    def __init__(
        self,
        n_nodes: int,
        coupling_strength: float = 1.0,
        dt: float = 0.01,
        omega_mean: float = 1.0,
        omega_std: float = 0.1,
        alpha_mean: float = 0.0,
        alpha_std: float = 0.05,  # Variabilité hardware
        delay_min: float = 0.05,
        delay_max: float = 0.2,
        adjacency_matrix: Optional[np.ndarray] = None,
        sparse_k: Optional[int] = None  # k plus proches voisins
    ):
        """
        Args:
            n_nodes: Nombre de nœuds
            coupling_strength: Force de couplage K
            dt: Pas de temps
            omega_mean/std: Paramètres de la distribution des fréquences
            alpha_mean/std: Paramètres de la distribution des frustrations
            delay_min/max: Bornes des délais de communication
            adjacency_matrix: Matrice d'adjacence. Si None, générée.
            sparse_k: Si spécifié, chaque nœud ne se connecte qu'à k voisins
        """
        self.n_nodes = n_nodes
        self.K = coupling_strength
        self.dt = dt
        
        # Créer les nœuds avec paramètres individuels
        self.nodes: List[KuramotoNode] = []
        for i in range(n_nodes):
            node = KuramotoNode(
                id=i,
                omega=np.random.normal(omega_mean, omega_std),
                alpha=np.random.normal(alpha_mean, alpha_std),  # Variabilité hardware !
                phase=np.random.uniform(0, 2 * np.pi)
            )
            self.nodes.append(node)
        
        # Matrice d'adjacence
        if adjacency_matrix is not None:
            self.adjacency = adjacency_matrix
        elif sparse_k is not None:
            self.adjacency = self._create_knn_adjacency(sparse_k)
        else:
            # Réseau complet
            self.adjacency = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        
        # Réseau de délais
        self.delay_network = DelayNetwork(
            n_nodes=n_nodes,
            state_dim=1,  # On stocke juste la phase
            dt=dt,
            delay_min=delay_min,
            delay_max=delay_max
        )
        
        # Diagnostics
        self.diagnostics = IntegrationDiagnostics()
        self.step_count = 0
        self.time = 0.0
        
        # Historique pour visualisation
        self.phase_history: List[np.ndarray] = []
        self.sync_history: List[float] = []
    
    def _create_knn_adjacency(self, k: int) -> np.ndarray:
        """Crée une matrice d'adjacence k-plus-proches-voisins circulaire."""
        adj = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for offset in range(1, k // 2 + 1):
                left = (i - offset) % self.n_nodes
                right = (i + offset) % self.n_nodes
                adj[i, left] = 1
                adj[i, right] = 1
        return adj
    
    def get_phases(self) -> np.ndarray:
        """Retourne les phases de tous les nœuds."""
        return np.array([node.phase for node in self.nodes])
    
    def set_phases(self, phases: np.ndarray):
        """Définit les phases de tous les nœuds."""
        for i, node in enumerate(self.nodes):
            node.phase = phases[i]
    
    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Calcule le paramètre d'ordre de Kuramoto.
        
        r = |1/N Σ e^(iθ_j)|
        
        r = 1: synchronisation parfaite
        r = 0: désordre complet
        
        Returns:
            Tuple (r, psi) où psi est la phase moyenne
        """
        phases = self.get_phases()
        complex_order = np.mean(np.exp(1j * phases))
        r = np.abs(complex_order)
        psi = np.angle(complex_order)
        return r, psi
    
    def derivative(self, t: float, phases: np.ndarray) -> np.ndarray:
        """
        Calcule dθ/dt pour tous les nœuds.
        
        C'est la fonction f(t, y) pour l'intégrateur.
        """
        d_phases = np.zeros(self.n_nodes)
        
        for i, node in enumerate(self.nodes):
            # Terme de fréquence naturelle
            d_phases[i] = node.omega
            
            # Terme de couplage
            coupling = 0.0
            n_neighbors = 0
            
            for j in range(self.n_nodes):
                if self.adjacency[i, j] > 0 and i != j:
                    # Récupérer la phase retardée de j
                    delayed_state = self.delay_network.get_delayed_state(j, i, t)
                    
                    if delayed_state is not None:
                        theta_j_delayed = delayed_state[0]
                    else:
                        # Fallback: utiliser la phase actuelle si pas assez d'historique
                        theta_j_delayed = phases[j]
                    
                    # Kuramoto-Sakaguchi avec frustration
                    coupling += np.sin(theta_j_delayed - phases[i] + node.alpha)
                    n_neighbors += 1
            
            if n_neighbors > 0:
                d_phases[i] += (self.K / n_neighbors) * coupling
        
        return d_phases
    
    def step(self) -> Tuple[float, float]:
        """
        Effectue un pas de simulation avec RK4.
        
        Returns:
            Tuple (r, error) - paramètre d'ordre et erreur d'intégration
        """
        phases = self.get_phases()
        
        # Mettre à jour le buffer de délais
        states = phases.reshape(-1, 1)
        self.delay_network.push_states(states, self.time)
        
        # Intégration RK4 avec monitoring
        new_phases, error = rk4_with_monitoring(
            f=self.derivative,
            y=phases,
            t=self.time,
            dt=self.dt,
            diagnostics=self.diagnostics,
            check_every=10,
            step_count=self.step_count
        )
        
        # Normaliser les phases dans [0, 2π]
        new_phases = new_phases % (2 * np.pi)
        
        self.set_phases(new_phases)
        self.time += self.dt
        self.step_count += 1
        
        # Calculer le paramètre d'ordre
        r, _ = self.compute_order_parameter()
        
        # Stocker l'historique
        self.phase_history.append(new_phases.copy())
        self.sync_history.append(r)
        
        return r, error
    
    def simulate(self, duration: float, verbose: bool = True) -> Dict:
        """
        Simule le réseau pendant une durée donnée.
        
        Args:
            duration: Durée en secondes
            verbose: Afficher la progression
        
        Returns:
            Dict avec les résultats
        """
        n_steps = int(duration / self.dt)
        
        if verbose:
            print(f"Simulation: {n_steps} pas, dt={self.dt*1000:.1f}ms")
            print(f"Délais: {self.delay_network.get_delay_stats()}")
        
        for step in range(n_steps):
            r, _ = self.step()
            
            if verbose and step % (n_steps // 10) == 0:
                print(f"  t={self.time:.2f}s | r={r:.4f}")
        
        # Résumé
        return {
            'final_r': self.sync_history[-1] if self.sync_history else 0,
            'max_r': max(self.sync_history) if self.sync_history else 0,
            'time': self.time,
            'diagnostics': self.diagnostics.summary()
        }
    
    def get_node_params(self) -> Dict:
        """Retourne les paramètres de tous les nœuds."""
        return {
            'omega': [n.omega for n in self.nodes],
            'alpha': [n.alpha for n in self.nodes],
            'alpha_std': np.std([n.alpha for n in self.nodes])
        }


# ==============================================================================
# Tests et Démonstration
# ==============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("TEST DU MODÈLE KURAMOTO-SAKAGUCHI AVEC DÉLAIS")
    print("=" * 60)
    
    # Créer un réseau de 10 nœuds
    network = KuramotoSakaguchiNetwork(
        n_nodes=10,
        coupling_strength=2.0,
        dt=0.01,
        omega_mean=1.0,
        omega_std=0.2,
        alpha_mean=0.05,  # Légère frustration (latence hardware)
        alpha_std=0.02,   # Variabilité entre nœuds
        delay_min=0.05,   # 50ms
        delay_max=0.15,   # 150ms
        sparse_k=4        # Chaque nœud connecté à 4 voisins
    )
    
    print(f"\nParamètres des nœuds:")
    params = network.get_node_params()
    print(f"  Fréquences ω: μ={np.mean(params['omega']):.3f}, σ={np.std(params['omega']):.3f}")
    print(f"  Frustrations α: μ={np.mean(params['alpha']):.3f}, σ={params['alpha_std']:.3f}")
    
    # Simuler
    print("\n--- Simulation ---")
    results = network.simulate(duration=5.0, verbose=True)
    
    print(f"\n--- Résultats ---")
    print(f"Paramètre d'ordre final: r = {results['final_r']:.4f}")
    print(f"Paramètre d'ordre max: r = {results['max_r']:.4f}")
    print(f"Diagnostics intégration: {results['diagnostics']}")
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Évolution du paramètre d'ordre
    ax1.plot(network.sync_history, linewidth=2, color='#27ae60')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Pas de temps')
    ax1.set_ylabel('Parametre d\'ordre r')
    ax1.set_title('Synchronisation Kuramoto-Sakaguchi avec delais')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Évolution des phases
    phase_history = np.array(network.phase_history)
    for i in range(network.n_nodes):
        ax2.plot(phase_history[:, i], alpha=0.7, linewidth=1)
    ax2.set_xlabel('Pas de temps')
    ax2.set_ylabel('Phase (rad)')
    ax2.set_title('Evolution des phases individuelles')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kuramoto_sakaguchi_test.png', dpi=150)
    print("\nImage sauvegardee: kuramoto_sakaguchi_test.png")
    plt.show()
    
    print("\n✅ Module kuramoto.py opérationnel")
