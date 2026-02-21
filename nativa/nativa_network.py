import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from neuron import LIFNeuron, NeuronPopulation
from stdp import SynapticMatrix, STDPConfig
from free_energy import VectorizedFreeEnergy
from event_encoder import RateEncoder, TemporalEncoder, DeltaEncoder

@dataclass
class NativaConfig:
    n_neurons: int = 200
    n_output_classes: int = 10
    connectivity: float = 0.2
    neuron_params: LIFNeuron = field(default_factory=LIFNeuron)
    stdp_config: STDPConfig = field(default_factory=STDPConfig)
    fe_learning_rate: float = 0.1
    fe_beta: float = 2.0
    encoder_type: str = "rate"
    n_encoding_steps: int = 20
    dt: float = 1.0
    seed: Optional[int] = None
    wta_inhibition: float = 10.0       
    weight_norm_target: float = 200.0   # Poids très forts
    use_kuramoto: bool = True          
    kuramoto_coupling: float = 2.0     
    kuramoto_omega: float = 1.0        
    kuramoto_dt: float = 0.01          
    use_adaptive_thresh: bool = True   
    thresh_increment: float = 1.0      
    use_rl: bool = True               
    eligibility_tau: float = 100.0      

    def __post_init__(self):
        assert self.n_neurons > 0

class NativaNetwork:
    """
    Réseau NATIVA v2 - Version Industrielle Déterministe
    """
    def __init__(self, config: Optional[NativaConfig] = None):
        self.config = config or NativaConfig()
        cfg = self.config

        self.neurons = NeuronPopulation(n_neurons=cfg.n_neurons, params=cfg.neuron_params, seed=cfg.seed)
        self.synapses = SynapticMatrix(n_neurons=cfg.n_neurons, config=cfg.stdp_config, connectivity=cfg.connectivity, seed=cfg.seed)
        self.free_energy = VectorizedFreeEnergy(n_agents=cfg.n_neurons, dim=1)
        
        # Init variables
        self._n_features = 0
        self.W_input = None
        self.trace_post = np.zeros(cfg.n_neurons)
        self._adaptive_thresh = np.zeros(cfg.n_neurons) 
        self._thresh_increment = cfg.thresh_increment    
        self._thresh_decay = 0.999      

        # Kuramoto
        if cfg.use_kuramoto:
            rng_k = np.random.default_rng(cfg.seed)
            self._kuramoto_phases = rng_k.uniform(0, 2 * np.pi, cfg.n_neurons)
            self._kuramoto_omega = rng_k.normal(cfg.kuramoto_omega, 0.1, cfg.n_neurons)
            self._kuramoto_K = cfg.kuramoto_coupling
            self._kuramoto_dt = cfg.kuramoto_dt
        else:
            self._kuramoto_phases = None

        rng = np.random.default_rng(cfg.seed)
        self.W_readout = rng.uniform(0, 0.1, size=(cfg.n_output_classes, cfg.n_neurons))

        self.fe_history = []
        self.step_count = 0
        self._fe_baseline = 1.0
        self._fe_alpha = 0.01
        self._fe_modulation = 1.0

    def _init_input_weights(self, n_features: int):
        self._n_features = n_features
        rng = np.random.default_rng(self.config.seed)
        # Init Poids Forts
        self.W_input = rng.uniform(0.5, 1.0, size=(self.config.n_neurons, n_features))
        row_sums = np.sum(self.W_input, axis=1, keepdims=True)
        self.W_input *= self.config.weight_norm_target / (row_sums + 1e-10)
        print(f"🔌 NATIVA : Input connecté (Canaux: {n_features})")

    def reset(self):
        """Réinitialise l'état interne du réseau (potentiels, seuils, FE baseline)."""
        cfg = self.config
        self.neurons.V[:] = cfg.neuron_params.V_reset
        self.trace_post[:] = 0.0
        if cfg.use_adaptive_thresh:
            self._adaptive_thresh[:] = 0.0
        self._fe_baseline = 1.0
        self._fe_modulation = 1.0
        self.fe_history = []
        self.step_count = 0
        self.temporal_surprise = []

    def _kuramoto_step(self):
        if self._kuramoto_phases is None: return
        phases = self._kuramoto_phases
        complex_order = np.mean(np.exp(1j * phases))
        R = np.abs(complex_order)
        Psi = np.angle(complex_order)
        d_phases = self._kuramoto_omega + self._kuramoto_K * R * np.sin(Psi - phases)
        self._kuramoto_phases = (phases + d_phases * self._kuramoto_dt) % (2 * np.pi)

    def _get_phase_coherence(self) -> np.ndarray:
        if self._kuramoto_phases is None: return np.ones(self.config.n_neurons)
        complex_order = np.mean(np.exp(1j * self._kuramoto_phases))
        Psi = np.angle(complex_order)
        return (1 + np.cos(self._kuramoto_phases - Psi)) / 2

    def feed(self, data: np.ndarray, mode: str = "train") -> Dict:
        """
        Mode Industriel : Encodage par Seuil (Threshold).
        Plus de hasard. Si signal > seuil, ça tire.
        """
        cfg = self.config
        learn = (mode == "train") 
        
        # --- 0. ENCODAGE DÉTERMINISTE (NOUVEAU) ---
        if data.ndim == 1:
            # Normalisation locale au cas où
            normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
            
            # SEUIL INDUSTRIEL : Tout ce qui dépasse 20% d'amplitude devient un spike (1.0)
            # Tout ce qui est en dessous devient 0.0
            # C'est radical et ça garantit l'activité.
            spike_train = (normalized > 0.2).astype(float).reshape(-1, 1)
            
        elif data.ndim == 2:
            spike_train = data
        else:
            raise ValueError("Format data incorrect")

        # --- 1. Init Poids ---
        n_features = spike_train.shape[-1]
        if self.W_input is None or self._n_features != n_features:
            self._init_input_weights(n_features)

        # --- 2. Préparation ---
        alpha_smooth = 0.2
        current_smooth_F = 0.0
        n_steps = spike_train.shape[0]
        total_spikes = 0
        all_internal_spikes = np.zeros(cfg.n_neurons, dtype=np.float64)
        spike_history = np.zeros((n_steps, cfg.n_neurons)) 

        # --- 3. RESET ---
        self.neurons.V[:] = cfg.neuron_params.V_reset 
        # Reset des seuils adaptatifs pour que le réseau soit frais à chaque écoute
        if cfg.use_adaptive_thresh:
            self._adaptive_thresh[:] = 0.0 
        self.temporal_surprise = [] 
        
        A_plus = cfg.stdp_config.A_plus * (0.5 + self._fe_modulation)

        # ======================================================================
        # BOUCLE TEMPORELLE
        # ======================================================================
        for t in range(n_steps):
            input_spikes = spike_train[t]

            # A. COURANT
            input_current = self.W_input @ input_spikes
            if cfg.use_adaptive_thresh:
                self._adaptive_thresh *= self._thresh_decay
            effective_current = input_current - self._adaptive_thresh

            # B. LIF
            current_time = self.step_count * cfg.dt + t * cfg.dt
            lif_spikes = self.neurons.step(
                effective_current, dt=cfg.dt, current_time=current_time
            )

            # C. WTA
            k_winners = int(cfg.wta_inhibition)
            alive = self.neurons.alive_mask
            candidates = effective_current * alive.astype(float)
            output_spikes = np.zeros(cfg.n_neurons)
            
            if np.any(candidates > 0):
                top_k_idx = np.argsort(candidates)[-k_winners:]
                for j in top_k_idx:
                    if candidates[j] > 0 and lif_spikes[j] > 0:
                        output_spikes[j] = 1.0

            # D. STATS
            spike_history[t, :] = output_spikes 
            total_spikes += int(np.sum(output_spikes))
            all_internal_spikes += output_spikes
            
            if cfg.use_adaptive_thresh and np.any(output_spikes):
                self._adaptive_thresh += output_spikes * self._thresh_increment

            # E. INFÉRENCE ACTIVE
            current_observation = output_spikes.reshape(-1, 1)
            step_F = self.free_energy.compute_elbo_batch(current_observation)
            mean_step_F = float(np.mean(step_F))

            current_smooth_F = (1 - alpha_smooth) * current_smooth_F + alpha_smooth * mean_step_F
            self.temporal_surprise.append(current_smooth_F)

            # F. STDP
            if learn and np.any(output_spikes):
                phase_mod = self._get_phase_coherence()
                spiking_idx = np.where(output_spikes > 0)[0]
                
                for j in spiking_idx:
                    self.W_input[j, :] += A_plus * phase_mod[j] * input_spikes
                    row_sum = self.W_input[j, :].sum()
                    if row_sum > 1e-10:
                        self.W_input[j, :] *= cfg.weight_norm_target / row_sum
                np.clip(self.W_input, 0, cfg.stdp_config.w_max, out=self.W_input)

            # G. KURAMOTO
            self._kuramoto_step()

        # SORTIE
        spike_rates = all_internal_spikes / max(n_steps, 1)
        mean_F = float(np.mean(self.temporal_surprise))
        
        self._fe_baseline = (
            (1 - self._fe_alpha) * self._fe_baseline + self._fe_alpha * mean_F
        )
        if self._fe_baseline > 0.001:
            surprise_ratio = mean_F / self._fe_baseline
            self._fe_modulation = float(1.0 / (1 + np.exp(-cfg.fe_beta * (surprise_ratio - 1.0))))

        # Debug Console
        print(f"   [NATIVA DEBUG] Spikes générés : {total_spikes}")

        return {
            'n_spikes': total_spikes,
            'spikes': spike_history,            
            'surprise': self.temporal_surprise
        }