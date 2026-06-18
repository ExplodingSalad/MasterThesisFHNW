"""
Comparative Experiment Runner
==============================
Runs WPT, PSO, and GA on the same IRP scenarios with multiple independent
runs per scenario, then performs statistical comparison.

This module is responsible only for orchestration: building the task list,
running the algorithms (in parallel via ProcessPoolExecutor), and
collecting results. All reporting (descriptive stats, statistical tests,
CSV/HTML export) lives in `reports.py`.

Usage:
    python run_comparison.py                                # uses ExperimentConfig dataclass defaults
    python run_comparison.py --n-runs 30 --n-instances 10   # override individual fields
    python run_comparison.py --levels simple intermediate
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass

from algorithms.genetic_algorithm import GAConfig, ga_optimize
from algorithms.particle_swarm_optimizer import pso_optimize, PSOConfig
from algorithms.walking_palmtree import WPTConfig, wpt_optimize
from irp_definition import make_irp_objective
from reports import (
    AlgoRunResult,
    analyse,
    export_csv,
    format_eta,
    generate_convergence_html,
    progress_bar,
)
from scenario_generator import generate_scenario


# ======================================================================
# Experiment configuration
# ======================================================================
@dataclass
class ExperimentConfig:
    """Controls the entire experimental setup."""
    levels: list[str]                         # scenario complexity levels
    n_runs: int = 15                          # independent runs per algorithm per scenario. NOTE: using more than 15 runs requires >24GB of RAM
    n_scenario_instances: int = 10          # random instances per level
    max_iter: int = 500                       # iterations for WPT
    pop_size: int = 30                       # population size for all algorithms
    n_roots_wpt: int = 30                     # roots per tree (WPT only)
    output_dir: str = "results"


# ======================================================================
# Fair-comparison helper
# ======================================================================
#
# WPT evaluates ~(n_trees * (n_roots + 1)) objectives per *exploration*
# pass, but its exploitation phase ALSO evaluates roots that drift out
# of the neighbourhood (frequent in practice). Empirically we observe
# ~2x the exploration estimate. The factor below is an empirical
# correction;
# ======================================================================
WPT_EXPLOITATION_OVERHEAD = 2.0   # empirical: actual evals/iter ≈ 2 × estimate

def _estimate_wpt_evals_per_iter(n_trees: int, n_roots: int) -> int:
    """Approximate evals per WPT iteration (excluding init).

    Per iter: (n_trees - 2) trees update + each has n_roots root evals
    (exploration phase), then exploitation may replace out-of-range
    roots (~n_trees * n_roots * 0.5 typical in practice).
    """
    active_trees = max(n_trees - 2, 1)
    base = active_trees * (1 + n_roots) + n_trees
    return int(base * WPT_EXPLOITATION_OVERHEAD)


# ======================================================================
# Parallelisable worker (top-level for pickle compatibility)
# ======================================================================
# ProcessPoolExecutor requires that the worker function and its arguments
# are picklable. Closures (like the objective function returned by
# make_irp_objective) are NOT picklable. So we pass the scenario seed
# and rebuild the objective inside each worker process.
# ======================================================================

def _run_single_task(task: dict) -> dict:
    """Execute one algorithm run in a worker process."""
    level = task["level"]
    inst_seed = task["inst_seed"]
    inst = task["inst"]
    run_idx = task["run_idx"]
    alg_name = task["alg_name"]
    alg_seed = task["alg_seed"]
    max_iter = task["max_iter"]
    pop_size = task["pop_size"]
    n_roots = task["n_roots"]

    # Rebuild scenario and objective inside the worker process
    scenario = generate_scenario(level=level, seed=inst_seed)
    objective, lb, ub = make_irp_objective(scenario)

    t0 = time.perf_counter()

    if alg_name == "WPT":
        cfg = WPTConfig(n_trees=pop_size, n_roots=n_roots,
                        max_iter=max_iter, seed=alg_seed)
        res = wpt_optimize(objective, lb, ub, cfg)
    elif alg_name == "PSO":
        cfg = PSOConfig(n_particles=pop_size, max_iter=max_iter,
                        seed=alg_seed)
        res = pso_optimize(objective, lb, ub, cfg)
    else:  # GA
        cfg = GAConfig(pop_size=pop_size, max_iter=max_iter,
                       seed=alg_seed)
        res = ga_optimize(objective, lb, ub, cfg)

    elapsed = time.perf_counter() - t0

    return {
        "algorithm": alg_name,
        "level": level,
        "instance": inst,
        "run": run_idx,
        "best_fitness": res.best_fitness,
        "elapsed_sec": elapsed,
        "n_evals": res.n_evals,
        "convergence": res.convergence_curve,
    }


# ======================================================================
# Run full experiment
# ======================================================================
def run_experiment(ecfg: ExperimentConfig) -> list[AlgoRunResult]:
    """Execute the full comparative experiment with parallel workers."""

    # ------------------------------------------------------------------
    # Phase 1: Print scenario parameters (sequential, once per instance)
    # ------------------------------------------------------------------
    import numpy as np  # local import keeps top-level lean

    for level_idx, level in enumerate(ecfg.levels):
        print(f"\n{'='*70}")
        print(f"  SCENARIO LEVEL: {level.upper()}  "
              f"({level_idx+1}/{len(ecfg.levels)})")
        print(f"{'='*70}")

        for inst in range(ecfg.n_scenario_instances):
            inst_seed = 1000 * (ecfg.levels.index(level) + 1) + inst
            scenario = generate_scenario(level=level, seed=inst_seed)
            objective, lb, ub = make_irp_objective(scenario)
            d = len(lb)

            print(f"\n  Instance {inst+1}/{ecfg.n_scenario_instances}  "
                  f"(d={d}, seed={inst_seed})")
            print(f"    Scenario parameters:")
            print(f"      Supplier:  warehouses={scenario.n_supplier_warehouses}, "
                  f"capacity/wh={scenario.supplier_warehouse_capacity:.0f}, "
                  f"total_cap={scenario.total_supplier_capacity:.0f}")
            print(f"      Products:  n={scenario.n_products}, "
                  f"sizes={np.array2string(scenario.product_sizes, precision=1, separator=', ')}, "
                  f"h_sup={np.array2string(scenario.supplier_holding_cost, precision=4, separator=', ')}, "
                  f"h_cust={np.array2string(scenario.customer_holding_cost, precision=4, separator=', ')}")
            print(f"      Vehicles:  types={scenario.n_vehicle_types}, "
                  f"per_type={scenario.vehicles_per_type}, "
                  f"capacity={np.array2string(scenario.vehicle_capacity, precision=1, separator=', ')}, "
                  f"cost/dist={np.array2string(scenario.vehicle_cost_per_dist, precision=2, separator=', ')}")
            print(f"      Customers: n={scenario.n_customers}, "
                  f"warehouses/cust={np.array2string(scenario.n_customer_warehouses, separator=', ')}, "
                  f"wh_cap={np.array2string(scenario.customer_warehouse_capacity, precision=1, separator=', ')}")
            print(f"      Demand:    ", end="")
            cons = scenario.daily_consumption
            if cons.size <= 10:
                print(f"consumption={np.array2string(cons, precision=1, separator=', ')}")
            else:
                print(f"shape={cons.shape}, "
                      f"range=[{cons.min():.0f}, {cons.max():.0f}], "
                      f"mean={cons.mean():.1f}")
            print(f"      Policy:    RS={scenario.replenishment_strategy} "
                  f"({'max-level' if scenario.replenishment_strategy == 1 else 'order-up-to'}), "
                  f"periods={scenario.n_periods}")

    # ------------------------------------------------------------------
    # Phase 2: Build all tasks
    # ------------------------------------------------------------------
    tasks: list[dict] = []

    for level in ecfg.levels:
        for inst in range(ecfg.n_scenario_instances):
            inst_seed = 1000 * (ecfg.levels.index(level) + 1) + inst

            # Compute fair iteration counts so PSO/GA's eval budget ≈ WPT's.
            wpt_evals_per_iter = _estimate_wpt_evals_per_iter(
                ecfg.pop_size, ecfg.n_roots_wpt
            )
            eval_ratio = max(1, wpt_evals_per_iter // ecfg.pop_size)
            pso_ga_iter = ecfg.max_iter * eval_ratio

            for run_idx in range(ecfg.n_runs):
                base_seed = inst_seed * 100 + run_idx
                algo_seeds = {
                    "WPT": base_seed * 3,
                    "PSO": base_seed * 3 + 1,
                    "GA":  base_seed * 3 + 2,
                }

                for alg_name in ["WPT", "PSO", "GA"]:
                    tasks.append({
                        "level": level,
                        "inst_seed": inst_seed,
                        "inst": inst,
                        "run_idx": run_idx,
                        "alg_name": alg_name,
                        "alg_seed": algo_seeds[alg_name],
                        "max_iter": ecfg.max_iter if alg_name == "WPT" else pso_ga_iter,
                        "pop_size": ecfg.pop_size,
                        "n_roots": ecfg.n_roots_wpt,
                    })

    total_tasks = len(tasks)
    n_workers = min(os.cpu_count() or 1, total_tasks)

    # ------------------------------------------------------------------
    # Phase 3: Execute all tasks in parallel
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RUNNING {total_tasks} tasks on {n_workers} CPU cores")
    print(f"  (PSO/GA iterations = {pso_ga_iter}, WPT iterations = {ecfg.max_iter}, "
          f"ratio = {eval_ratio}x)")
    print(f"{'='*70}\n")

    all_results: list[AlgoRunResult] = []
    completed = 0
    experiment_start = time.perf_counter()
    HEARTBEAT_SEC = 15.0   # liveness ping while no task has finished, as it may be weird to not have feedback in console

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_task, t): t for t in tasks}
        remaining = set(futures.keys())

        # Loop with a timeout so we can emit heartbeats even before any
        # task completes (important for huge tasks where the first
        # completion may be many minutes away).
        while remaining:
            done_now, remaining = wait(
                remaining, timeout=HEARTBEAT_SEC,
                return_when=FIRST_COMPLETED,
            )

            if not done_now:
                elapsed = time.perf_counter() - experiment_start
                bar = progress_bar(completed, total_tasks)
                print(f"  {bar}  [{completed:>5}/{total_tasks}]  "
                      f"... working, no completion in last "
                      f"{HEARTBEAT_SEC:.0f}s "
                      f"(elapsed {format_eta(elapsed)})",
                      flush=True)
                continue

            for future in done_now:
                completed += 1
                result = future.result()

                all_results.append(AlgoRunResult(
                    algorithm=result["algorithm"],
                    level=result["level"],
                    instance=result["instance"],
                    run=result["run"],
                    best_fitness=result["best_fitness"],
                    elapsed_sec=result["elapsed_sec"],
                    n_evals=result["n_evals"],
                    convergence=result["convergence"],
                ))

                elapsed = time.perf_counter() - experiment_start
                rate = elapsed / completed
                eta = rate * (total_tasks - completed)
                eta_str = format_eta(eta)

                bar = progress_bar(completed, total_tasks)
                r = result
                # One line per completed task so progress is visible in
                # IDE run-consoles (PyCharm, VS Code, etc.) which don't
                # render carriage-return overwrites correctly.
                print(f"  {bar}  [{completed:>5}/{total_tasks}]  "
                      f"{r['level']}/inst{r['instance']}/run{r['run']} "
                      f"{r['algorithm']}={r['best_fitness']:,.0f} "
                      f"({r['elapsed_sec']:.1f}s)  "
                      f"[ETA: {eta_str}]",
                      flush=True)

    total_elapsed = time.perf_counter() - experiment_start
    print(f"\n  All {total_tasks} tasks completed in "
          f"{format_eta(total_elapsed)} using {n_workers} workers")

    return all_results


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="IRP Algorithm Comparison")
    parser.add_argument("--levels", nargs="+", default=None,
                        help="Scenario levels to test (overrides dataclass)")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Override n_runs from ExperimentConfig")
    parser.add_argument("--n-instances", type=int, default=None,
                        help="Override n_scenario_instances from ExperimentConfig")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Override max_iter from ExperimentConfig")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for output files")
    args = parser.parse_args()

    # ExperimentConfig dataclass defaults (top of this file) are the single
    # source of truth. CLI flags override individual fields if provided.
    ecfg = ExperimentConfig(
        levels=args.levels or ["basic", "simple", "intermediate",
                               "advanced", "complex"],
        output_dir=args.output_dir,
    )

    if args.n_runs is not None:
        ecfg.n_runs = args.n_runs
    if args.n_instances is not None:
        ecfg.n_scenario_instances = args.n_instances
    if args.max_iter is not None:
        ecfg.max_iter = args.max_iter

    print(f"  Experiment configuration:")
    print(f"    Levels           : {ecfg.levels}")
    print(f"    Instances/level  : {ecfg.n_scenario_instances}")
    print(f"    Runs/instance    : {ecfg.n_runs}")
    print(f"    Iterations (WPT) : {ecfg.max_iter}")
    print(f"    Population size  : {ecfg.pop_size}")
    print(f"    CPU cores        : {os.cpu_count()}")

    results = run_experiment(ecfg)
    analyse(results, ecfg)
    export_csv(results, ecfg.output_dir)
    generate_convergence_html(results, ecfg.output_dir)

    print(f"\n{'='*80}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
