"""
Walking Palm Tree (WPT) Optimizer
=================================
Implementation based on:
    Zitouni et al. (2024) - "The Walking Palm Tree algorithm:
    A new Metaheuristic Algorithm for Solving Optimization Problems"

Maps from paper notation to code:
    n               - cfg.n_trees             (palm trees / clusters)
    m               - cfg.n_roots             (roots per tree)
    Δ               - cfg.max_iter
    ε               - cfg.epsilon              (neighbourhood size)
    σ1, σ2          - cfg.sigma1, cfg.sigma2  (Eq. 8 scale schedule)
    ρ               - cfg.rho                 (reproduction probability)
    β               - cfg.levy_beta           (Lévy flight power-law)
    S_{i,1}         - trees[i]                (cluster centroid)
    S_{i,2..m+1}    - roots[i, 0..m-1]        (cluster roots)
    A, A1, A2, A3   - Heron-formula triangle areas (Step 2 shedding test)
    T               - Eq. 7 target location

Performance Optimizations
-------------------------
The naive per-element implementation has been progressively vectorised
without changing the algorithm's mathematical behaviour OR the order of
random-number draws (so seeded runs reproduce bit-exactly).

  OPT-1: Triangle area uses dot-product formula instead of Heron's
         (3 dot products vs. 3 norms + Heron's).

  OPT-2: Per-iteration constants (beta_param, t1, t2, scale) are
         precomputed once per outer iteration instead of per tree.

  OPT-3: Trees and roots stored as numpy arrays (n,d) and (n,m,d),
         not Python list-of-lists.

  OPT-4: Initial root generation uses a single vectorised numpy call
         (`_generate_roots_batch`) instead of m sequential per-root calls.

  OPT-5: Out-of-range root detection in the exploitation phase uses one
         batched `np.linalg.norm` over all m roots.

  OPT-6: In-place numpy operations (np.clip(..., out=), np.maximum(..., out=))
         to reduce temporary array allocations.

  OPT-7: Top-of-iteration triangle areas A(△ S_{i,1} S_{fb,1} S_{sb,1})
         for all n trees are computed in a single vectorised batch
         (`_areas_n_shared_p2p3`) instead of per-tree.

  OPT-8: The per-root inner j-loop's three triangle-area calls
         (A1, A2, A3) are replaced by THREE vectorised batches
         (`_areas_m_shared_p1p2`) operating on all m roots at once.
         This is the dominant per-iteration cost driver and the largest
         speedup. RNG draws inside the j-loop remain in their original
         order to preserve bit-exact reproducibility.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class WPTConfig:
    """Tuneable parameters of the WPT algorithm (defaults from Table 1)."""
    n_trees: int = 30                # n  – palm trees (clusters)
    n_roots: int = 10                # m  – roots per tree
    max_iter: int = 500              # Δ  – maximum iterations
    epsilon: float = 1e-4            # ε  – neighbourhood size
    sigma1: float = 1.0              # σ₁ – initial scale value
    sigma2: float = 1e-6             # σ₂ – final scale value
    rho: float = 1e-6                # ρ  – reproduction probability
    levy_beta: float = 1.5           # β  – Lévy power-law (1 < β < 2)
    seed: Optional[int] = None       # reproducibility


# ---------------------------------------------------------------------------
# Triangle-area helpers
# ---------------------------------------------------------------------------
# Identity used:  Area = ½ · √(|AB|² · |AC|² − (AB · AC)²)
# requires only 3 dot products instead of 3 norms + Heron's formula.

def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Area of a single triangle (a, b, c) in ℝ^d via dot products. (OPT-1)"""
    ab = b - a
    ac = c - a
    val = np.dot(ab, ab) * np.dot(ac, ac) - np.dot(ab, ac) ** 2
    return 0.5 * math.sqrt(max(val, 0.0))


def _areas_n_shared_p2p3(
    verts: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
    """
    Areas of n triangles (verts[i], p2, p3).  (OPT-7)
        verts : (n, d)  per-triangle first vertex
        p2, p3 : (d,)   shared second/third vertices
    Returns (n,) array of areas.
    """
    ab = p2[None, :] - verts                      # (n, d)
    ac = p3[None, :] - verts                      # (n, d)
    dot_ab = np.einsum("ij,ij->i", ab, ab)        # (n,)
    dot_ac = np.einsum("ij,ij->i", ac, ac)        # (n,)
    dot_ab_ac = np.einsum("ij,ij->i", ab, ac)     # (n,)
    val = dot_ab * dot_ac - dot_ab_ac * dot_ab_ac
    return 0.5 * np.sqrt(np.maximum(val, 0.0))


def _areas_m_shared_p1p2(
    p1: np.ndarray, p2: np.ndarray, p3_batch: np.ndarray
) -> np.ndarray:
    """
    Areas of m triangles (p1, p2, p3_batch[j]).  (OPT-8)
        p1, p2 : (d,) shared first/second vertices
        p3_batch : (m, d) varied third vertex
    Returns (m,) array of areas.
    """
    ab = p2 - p1                                    # (d,)
    ac = p3_batch - p1[None, :]                     # (m, d)
    dot_ab = float(np.dot(ab, ab))                  # scalar
    dot_ac = np.einsum("ij,ij->i", ac, ac)          # (m,)
    dot_ab_ac = ac @ ab                             # (m,)
    val = dot_ab * dot_ac - dot_ab_ac * dot_ab_ac
    return 0.5 * np.sqrt(np.maximum(val, 0.0))


# ---------------------------------------------------------------------------
# Lévy flight  (Equation 11)
# ---------------------------------------------------------------------------
def _levy_step(d: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = rng.normal(0, sigma_u, size=d)
    v = rng.normal(0, 1, size=d)
    return u / (np.abs(v) ** (1 / beta))


# ---------------------------------------------------------------------------
# Batched root initialisation around a centroid  (Equation 3)
# ---------------------------------------------------------------------------
def _generate_roots_batch(
    tree_pos: np.ndarray, m: int, eps: float,
    L: np.ndarray, U: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Generate m root positions around tree_pos. Shape (m, d).  (OPT-4)"""
    d = len(tree_pos)
    U_prime = rng.uniform(0, 1, size=(m, d))
    roots = tree_pos[None, :] + eps * (2 * U_prime - 1)
    np.clip(roots, L[None, :], U[None, :], out=roots)
    return roots


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class WPTResult:
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: list[float] = field(default_factory=list)
    n_evals: int = 0


# ---------------------------------------------------------------------------
# Core WPT optimiser
# ---------------------------------------------------------------------------
def wpt_optimize(
    objective: Callable[[np.ndarray], float],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    cfg: WPTConfig | None = None,
) -> WPTResult:
    """Minimise *objective* over the box [lower_bounds, upper_bounds]."""
    if cfg is None:
        cfg = WPTConfig()

    rng = np.random.default_rng(cfg.seed)

    d = len(lower_bounds)
    L = lower_bounds.astype(float)
    U = upper_bounds.astype(float)
    n = cfg.n_trees
    m = cfg.n_roots
    Delta = cfg.max_iter
    eps = cfg.epsilon

    # τ – shedding threshold  (Eq. 6)
    mu0, mu1, mu2 = -1.6, 1.08, 1.00
    tau = math.exp(mu0) * (d ** mu1) * (eps ** mu2)

    # γ for t2  (Eq. 8)
    gamma_val = math.log((0.5 - cfg.sigma2) / (cfg.sigma1 - cfg.sigma2)) / math.log(0.5)
    beta_param = math.log(0.5) / ((Delta / 2) ** 2)         # OPT-2

    n_evals = 0

    # ------------------------------------------------------------------
    # Initialisation  (Algorithm 1, lines 1-6)
    # ------------------------------------------------------------------
    trees = rng.uniform(L, U, size=(n, d))                  # OPT-3
    tree_fits = np.array([objective(trees[i]) for i in range(n)])
    n_evals += n

    roots = np.empty((n, m, d))
    root_fits = np.empty((n, m))
    for i in range(n):
        roots[i] = _generate_roots_batch(trees[i], m, eps, L, U, rng)  # OPT-4
        for j in range(m):
            root_fits[i, j] = objective(roots[i, j])
            n_evals += 1

    convergence: list[float] = []

    # ------------------------------------------------------------------
    # Main loop  (Algorithm 1, line 7)
    # ------------------------------------------------------------------
    for t in range(1, Delta + 1):
        # Per-iteration constants (OPT-2)
        ratio = t / Delta
        t2 = ((1 - ratio) ** gamma_val) * (cfg.sigma1 - cfg.sigma2) + cfg.sigma2
        t1 = 1 - math.exp(beta_param * (t2 ** 2))
        scale = t1 * t2

        # Top-2 trees fb, sb
        sorted_idx = np.argsort(tree_fits)
        fb = int(sorted_idx[0])
        sb = int(sorted_idx[1])
        fb_pos = trees[fb]
        sb_pos = trees[sb]
        fY = tree_fits[fb]
        fZ = tree_fits[sb]

        # OPT-7: A(△ S_{i,1} S_{fb,1} S_{sb,1}) for all n trees in one call.
        # The areas are read inside the per-tree loop using the original
        # (pre-update) tree positions, exactly as the paper specifies.
        A_all = _areas_n_shared_p2p3(trees, fb_pos, sb_pos)

        # ------------------------- Exploration  (lines 8-26)
        for i in range(n):
            if i == fb or i == sb:
                continue

            X = trees[i]
            fX = tree_fits[i]
            A = A_all[i]

            # Eq. 7 – target location T (no RNG)
            delta1 = abs(fX - fY)
            delta2 = abs(fX - fZ)
            diff_y = np.abs(X - fb_pos)
            diff_z = np.abs(X - sb_pos)
            np.maximum(diff_y, 1e-30, out=diff_y)            # OPT-6
            np.maximum(diff_z, 1e-30, out=diff_z)
            theta = np.arctan(delta1 / diff_y)
            theta_prime = np.arctan(delta2 / diff_z)
            sum_theta = theta + theta_prime
            np.maximum(sum_theta, 1e-30, out=sum_theta)
            Phi = np.minimum(theta, theta_prime) / sum_theta
            Phi_prime = np.maximum(theta, theta_prime) / sum_theta
            T_target = Phi * fb_pos + Phi_prime * sb_pos

            # Tree update – Eq. 8 (RNG order preserved exactly)
            alpha_rand = rng.uniform()
            G = rng.normal(0, 1, size=d)
            new_tree = X + scale * (T_target - X) + alpha_rand * G
            np.clip(new_tree, L, U, out=new_tree)            # OPT-6
            trees[i] = new_tree
            tree_fits[i] = objective(new_tree)
            n_evals += 1

            # OPT-8: Batched A1, A2, A3 for all m roots of cluster i.
            #   A1[j] = area(trees[i],   fb_pos, roots[i, j])
            #   A2[j] = area(trees[i],   sb_pos, roots[i, j])
            #   A3[j] = area(fb_pos,     sb_pos, roots[i, j])
            roots_i = roots[i]
            new_X = trees[i]   # post-update tree, as per paper's Step 2
            A1_arr = _areas_m_shared_p1p2(new_X, fb_pos, roots_i)
            A2_arr = _areas_m_shared_p1p2(new_X, sb_pos, roots_i)
            A3_arr = _areas_m_shared_p1p2(fb_pos, sb_pos, roots_i)
            need_replace = np.abs(A - (A1_arr + A2_arr + A3_arr)) > tau

            # Per-j conditional update – RNG ORDER must match the original
            # (one rng.uniform OR one rng.normal per j). Cannot batch.
            for j in range(m):
                if need_replace[j]:
                    # Shed root: Eq. 3
                    U_prime = rng.uniform(0, 1, size=d)
                    new_root = trees[i] + eps * (2 * U_prime - 1)
                    np.clip(new_root, L, U, out=new_root)    # OPT-6
                else:
                    # Update root: Eq. 8 (uses same T, alpha as tree update)
                    new_root = (
                        roots_i[j]
                        + scale * (T_target - roots_i[j])
                        + alpha_rand * rng.normal(0, 1, size=d)
                    )
                    np.clip(new_root, L, U, out=new_root)    # OPT-6
                roots[i, j] = new_root
                root_fits[i, j] = objective(new_root)
                n_evals += 1

        # ------------------------- Exploitation  (lines 27-35)
        for i in range(n):
            # ω – best element in cluster i
            cluster_fits = np.empty(m + 1)                   # OPT-3
            cluster_fits[0] = tree_fits[i]
            cluster_fits[1:] = root_fits[i]
            omega = int(np.argmin(cluster_fits))

            # Swap S_{i,1} ↔ S_{i,ω} if the best is a root
            if omega > 0:
                trees[i], roots[i, omega - 1] = (
                    roots[i, omega - 1].copy(), trees[i].copy(),
                )
                tree_fits[i], root_fits[i, omega - 1] = (
                    root_fits[i, omega - 1], tree_fits[i],
                )

            # OPT-5: Vectorised neighbourhood check; replace stragglers.
            dists = np.linalg.norm(roots[i] - trees[i][None, :], axis=1)
            out_of_range = dists > eps
            n_replace = int(out_of_range.sum())
            if n_replace > 0:
                new_roots = _generate_roots_batch(
                    trees[i], n_replace, eps, L, U, rng,
                )
                idx_replace = np.where(out_of_range)[0]
                for k, idx_r in enumerate(idx_replace):
                    roots[i, idx_r] = new_roots[k]
                    root_fits[i, idx_r] = objective(new_roots[k])
                    n_evals += 1

        # ------------------------- Reproduction  (lines 36-43)
        rho0 = rng.uniform()
        if rho0 <= cfg.rho:
            # Roulette-wheel selection over tree fitness (Eq. 10)
            shifted = tree_fits - tree_fits.min() + 1e-12
            probs = shifted / shifted.sum()
            p = rng.choice(n, p=probs)

            # Replace S_{p,1} via Lévy flight (Eq. 11)
            levy = _levy_step(d, cfg.levy_beta, rng)
            new_tree = trees[p] + levy
            np.clip(new_tree, L, U, out=new_tree)            # OPT-6
            trees[p] = new_tree
            tree_fits[p] = objective(new_tree)
            n_evals += 1

            # Replace its m roots (Eq. 3) – batched (OPT-4)
            new_roots = _generate_roots_batch(trees[p], m, eps, L, U, rng)
            roots[p] = new_roots
            for j in range(m):
                root_fits[p, j] = objective(new_roots[j])
                n_evals += 1

        convergence.append(float(tree_fits.min()))

    # ------------------------------------------------------------------
    # Best solution
    # ------------------------------------------------------------------
    best_idx = int(np.argmin(tree_fits))
    return WPTResult(
        best_position=trees[best_idx].copy(),
        best_fitness=tree_fits[best_idx],
        convergence_curve=convergence,
        n_evals=n_evals,
    )
