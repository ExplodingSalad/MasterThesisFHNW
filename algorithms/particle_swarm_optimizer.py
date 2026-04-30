"""
Particle Swarm Optimization (PSO)
==================================
Standard PSO with inertia weight, matching the interface of wpt_optimize()
for fair comparison.

References:
    Kennedy & Eberhart (1995) – original PSO
    Shi & Eberhart (1998) – inertia weight variant

Parameters follow Table 1 of Zitouni et al. (2024):
    inertia w = 0.3, c1 = 1, c2 = 1
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class PSOConfig:
    """PSO hyper-parameters."""
    n_particles: int = 30
    max_iter: int = 500
    w: float = 0.3          # inertia weight
    c1: float = 1.0         # cognitive coefficient
    c2: float = 1.0         # social coefficient
    seed: Optional[int] = None


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: list[float] = field(default_factory=list)
    n_evals: int = 0


def pso_optimize(
    objective: Callable[[np.ndarray], float],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    cfg: PSOConfig | None = None,
) -> PSOResult:
    """
    Minimise *objective* using standard PSO with inertia weight.
    """
    if cfg is None:
        cfg = PSOConfig()

    rng = np.random.default_rng(cfg.seed)
    d = len(lower_bounds)
    L = lower_bounds.astype(float)
    U = upper_bounds.astype(float)
    n = cfg.n_particles
    n_evals = 0

    # --- Initialisation ---
    positions = rng.uniform(L, U, size=(n, d))
    v_max = (U - L) * 0.2
    velocities = rng.uniform(-v_max, v_max, size=(n, d))

    # Evaluate initial population
    fitness = np.array([objective(positions[i]) for i in range(n)])
    n_evals += n

    # Personal best
    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    # Global best
    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]

    convergence: list[float] = []

    # --- Main loop ---
    for t in range(cfg.max_iter):
        r1 = rng.uniform(0, 1, size=(n, d))
        r2 = rng.uniform(0, 1, size=(n, d))

        # Update velocity
        velocities = (
            cfg.w * velocities
            + cfg.c1 * r1 * (pbest_pos - positions)
            + cfg.c2 * r2 * (gbest_pos - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)

        # Update position
        positions = positions + velocities
        positions = np.clip(positions, L, U)

        # Evaluate
        for i in range(n):
            fit = objective(positions[i])
            n_evals += 1

            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = positions[i].copy()

                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = positions[i].copy()

        convergence.append(gbest_fit)

    return PSOResult(
        best_position=gbest_pos,
        best_fitness=gbest_fit,
        convergence_curve=convergence,
        n_evals=n_evals,
    )