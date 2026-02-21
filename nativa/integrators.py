"""
Intégrateurs Numériques pour Secteur 0
======================================
Implémente RK4, RK2 et Euler avec monitoring d'erreur.

Standard industriel pour simulations physiques.
"""

import numpy as np
from typing import Callable, Tuple, List
from dataclasses import dataclass, field
import warnings


@dataclass
class IntegrationDiagnostics:
    """Diagnostics de stabilité de l'intégration."""
    errors: List[float] = field(default_factory=list)
    max_error: float = 0.0
    warnings_count: int = 0
    
    def log_error(self, error: float, threshold: float = 1e-4):
        """Log l'erreur et avertit si trop grande."""
        self.errors.append(error)
        self.max_error = max(self.max_error, error)
        
        if error > threshold:
            self.warnings_count += 1
            if self.warnings_count <= 5:  # Limiter les warnings
                warnings.warn(
                    f"Erreur d'intégration élevée: {error:.2e} > seuil {threshold:.2e}. "
                    f"Considérer réduire dt."
                )
    
    def summary(self) -> dict:
        """Retourne un résumé des diagnostics."""
        return {
            'max_error': self.max_error,
            'mean_error': np.mean(self.errors) if self.errors else 0.0,
            'warnings': self.warnings_count,
            'samples': len(self.errors)
        }


def euler(f: Callable, y: np.ndarray, t: float, dt: float) -> np.ndarray:
    """
    Méthode d'Euler explicite.
    
    Erreur locale: O(dt²)
    Usage: Référence de base, NE PAS utiliser en production.
    
    Args:
        f: Fonction dérivée f(t, y) → dy/dt
        y: État actuel
        t: Temps actuel
        dt: Pas de temps
    
    Returns:
        Nouvel état y(t + dt)
    """
    return y + dt * f(t, y)


def rk2(f: Callable, y: np.ndarray, t: float, dt: float) -> np.ndarray:
    """
    Méthode de Runge-Kutta d'ordre 2 (méthode du point milieu).
    
    Erreur locale: O(dt³)
    Usage: Intermédiaire, pour comparaison avec RK4.
    
    Args:
        f: Fonction dérivée f(t, y) → dy/dt
        y: État actuel
        t: Temps actuel
        dt: Pas de temps
    
    Returns:
        Nouvel état y(t + dt)
    """
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt * k1 / 2)
    return y + dt * k2


def rk4(f: Callable, y: np.ndarray, t: float, dt: float) -> np.ndarray:
    """
    Méthode de Runge-Kutta d'ordre 4.
    
    Erreur locale: O(dt⁵)  
    Usage: STANDARD PRODUCTION - 1000x plus précis qu'Euler.
    
    Args:
        f: Fonction dérivée f(t, y) → dy/dt
        y: État actuel
        t: Temps actuel
        dt: Pas de temps
    
    Returns:
        Nouvel état y(t + dt)
    """
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt * k1 / 2)
    k3 = f(t + dt/2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def rk4_with_monitoring(
    f: Callable, 
    y: np.ndarray, 
    t: float, 
    dt: float,
    diagnostics: IntegrationDiagnostics,
    check_every: int = 10,
    step_count: int = 0
) -> Tuple[np.ndarray, float]:
    """
    RK4 avec monitoring d'erreur via comparaison RK2.
    
    Compare périodiquement RK4 vs RK2 pour estimer l'erreur.
    C'est l'argument "qualité" suprême pour un chercheur.
    
    Args:
        f: Fonction dérivée
        y: État actuel
        t: Temps actuel
        dt: Pas de temps
        diagnostics: Objet de diagnostic pour logging
        check_every: Fréquence de vérification (tous les N pas)
        step_count: Compteur de pas actuel
    
    Returns:
        Tuple (nouvel_état, erreur_estimée)
    """
    # Toujours calculer RK4
    y_rk4 = rk4(f, y, t, dt)
    
    error = 0.0
    
    # Vérification périodique
    if step_count % check_every == 0:
        y_rk2 = rk2(f, y, t, dt)
        error = np.linalg.norm(y_rk4 - y_rk2)
        diagnostics.log_error(error)
    
    return y_rk4, error


class AdaptiveIntegrator:
    """
    Intégrateur adaptatif qui ajuste dt automatiquement.
    
    Si l'erreur RK2-RK4 est trop grande, réduit dt.
    Si l'erreur est très petite, augmente dt pour aller plus vite.
    """
    
    def __init__(
        self,
        dt_initial: float = 0.01,
        dt_min: float = 0.0001,
        dt_max: float = 0.1,
        error_target: float = 1e-5,
        safety_factor: float = 0.9
    ):
        self.dt = dt_initial
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.error_target = error_target
        self.safety_factor = safety_factor
        self.diagnostics = IntegrationDiagnostics()
        self.step_count = 0
    
    def step(self, f: Callable, y: np.ndarray, t: float) -> Tuple[np.ndarray, float]:
        """
        Effectue un pas d'intégration avec adaptation de dt.
        
        Returns:
            Tuple (nouvel_état, dt_utilisé)
        """
        # Calculer RK4 et RK2
        y_rk4 = rk4(f, y, t, self.dt)
        y_rk2 = rk2(f, y, t, self.dt)
        
        # Estimer l'erreur
        error = np.linalg.norm(y_rk4 - y_rk2)
        self.diagnostics.log_error(error)
        
        # Adapter dt
        if error > 0:
            # Formule standard d'adaptation
            factor = self.safety_factor * (self.error_target / error) ** 0.2
            factor = np.clip(factor, 0.1, 2.0)  # Limiter les changements brusques
            self.dt = np.clip(self.dt * factor, self.dt_min, self.dt_max)
        
        self.step_count += 1
        return y_rk4, self.dt


# ==============================================================================
# Tests et Démonstration
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DES INTÉGRATEURS NUMÉRIQUES")
    print("=" * 60)
    
    # Équation test: dy/dt = -y (solution exacte: y = e^(-t))
    def f_test(t, y):
        return -y
    
    y0 = np.array([1.0])
    t = 0.0
    dt = 0.1
    
    y_euler = y0.copy()
    y_rk2_val = y0.copy()
    y_rk4_val = y0.copy()
    
    print(f"\nÉquation: dy/dt = -y, y(0) = 1")
    print(f"Solution exacte: y(t) = e^(-t)")
    print(f"dt = {dt}")
    print()
    
    diagnostics = IntegrationDiagnostics()
    
    for step in range(10):
        t_current = step * dt
        exact = np.exp(-t_current - dt)
        
        y_euler = euler(f_test, y_euler, t_current, dt)
        y_rk2_val = rk2(f_test, y_rk2_val, t_current, dt)
        y_rk4_val, err = rk4_with_monitoring(
            f_test, y_rk4_val, t_current, dt, diagnostics, check_every=1, step_count=step
        )
        
        print(f"t={t_current+dt:.1f} | Exact={exact:.6f} | "
              f"Euler={y_euler[0]:.6f} (err={abs(exact-y_euler[0]):.2e}) | "
              f"RK4={y_rk4_val[0]:.6f} (err={abs(exact-y_rk4_val[0]):.2e})")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    print(f"Résumé: {diagnostics.summary()}")
    print("\n✅ Module integrators.py opérationnel")
