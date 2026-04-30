"""
Genetic Algorithm (GA) – Real-valued encoding
===============================================
Standard GA with tournament selection, blend crossover (BLX-α),
and Gaussian mutation, matching the interface of wpt_optimize()
for fair comparison.

References:
    Holland (1992) – Genetic Algorithms
    Eshelman & Schaffer (1993) – BLX-α crossover
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class GAConfig:
    # GA parameters
    pop_size: int = 30         # population size
    max_iter: int = 500        # generations
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1   # fraction of range
    tournament_size: int = 3
    blx_alpha: float = 0.5       # BLX-α parameter
    elitism: int = 2             # number of elite individuals preserved
    seed: Optional[int] = None


@dataclass
class GAResult:
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: list[float] = field(default_factory=list)
    n_evals: int = 0


def ga_optimize(
    objective: Callable[[np.ndarray], float],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    cfg: GAConfig | None = None,
) -> GAResult:
    """
    Minimise *objective* using a real-valued GA.
    """
    if cfg is None:
        cfg = GAConfig()

    rng = np.random.default_rng(cfg.seed)
    d = len(lower_bounds)
    L = lower_bounds.astype(float)
    U = upper_bounds.astype(float)
    ranges = U - L
    n = cfg.pop_size
    n_evals = 0

    # --- Initialisation ---
    population = rng.uniform(L, U, size=(n, d))
    fitness = np.array([objective(population[i]) for i in range(n)])
    n_evals += n

    best_idx = int(np.argmin(fitness))
    gbest_pos = population[best_idx].copy()
    gbest_fit = fitness[best_idx]

    convergence: list[float] = []

    # --- Main loop ---
    for gen in range(cfg.max_iter):
        new_pop = np.empty_like(population)
        new_fit = np.empty(n)

        # Elitism: carry over the best individuals
        elite_idx = np.argsort(fitness)[: cfg.elitism]
        for i, ei in enumerate(elite_idx):
            new_pop[i] = population[ei].copy()
            new_fit[i] = fitness[ei]

        # Fill remaining slots
        idx = cfg.elitism
        while idx < n:
            # Tournament selection – parent 1
            p1 = _tournament(fitness, cfg.tournament_size, rng)
            # Tournament selection – parent 2
            p2 = _tournament(fitness, cfg.tournament_size, rng)

            child1 = population[p1].copy()
            child2 = population[p2].copy()

            # BLX-α crossover
            if rng.uniform() < cfg.crossover_rate:
                child1, child2 = _blx_crossover(
                    population[p1], population[p2],
                    cfg.blx_alpha, L, U, rng,
                )

            # Gaussian mutation
            child1 = _mutate(child1, cfg.mutation_rate, cfg.mutation_sigma, ranges, L, U, rng)
            child2 = _mutate(child2, cfg.mutation_rate, cfg.mutation_sigma, ranges, L, U, rng)

            new_pop[idx] = child1
            new_fit[idx] = objective(child1)
            n_evals += 1
            idx += 1

            if idx < n:
                new_pop[idx] = child2
                new_fit[idx] = objective(child2)
                n_evals += 1
                idx += 1

        population = new_pop
        fitness = new_fit

        # Update global best
        gen_best = int(np.argmin(fitness))
        if fitness[gen_best] < gbest_fit:
            gbest_fit = fitness[gen_best]
            gbest_pos = population[gen_best].copy()

        convergence.append(gbest_fit)

    return GAResult(
        best_position=gbest_pos,
        best_fitness=gbest_fit,
        convergence_curve=convergence,
        n_evals=n_evals,
    )


# --- GA operators ---

def _tournament(
    fitness: np.ndarray, k: int, rng: np.random.Generator
) -> int:
    """Tournament selection: pick k random individuals, return the best."""
    candidates = rng.choice(len(fitness), size=k, replace=False)
    return int(candidates[np.argmin(fitness[candidates])])


def _blx_crossover(
    p1: np.ndarray, p2: np.ndarray,
    alpha: float,
    L: np.ndarray, U: np.ndarray,
    rng: np.random.Generator,
):
    """BLX-α crossover producing two children."""
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    span = hi - lo
    child1 = rng.uniform(lo - alpha * span, hi + alpha * span)
    child2 = rng.uniform(lo - alpha * span, hi + alpha * span)
    child1 = np.clip(child1, L, U)
    child2 = np.clip(child2, L, U)
    return child1, child2


def _mutate(
    ind: np.ndarray,
    rate: float, sigma: float,
    ranges: np.ndarray,
    L: np.ndarray, U: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-gene Gaussian mutation."""
    mask = rng.uniform(size=len(ind)) < rate
    noise = rng.normal(0, sigma * ranges, size=len(ind))
    ind = ind + mask * noise
    return np.clip(ind, L, U)