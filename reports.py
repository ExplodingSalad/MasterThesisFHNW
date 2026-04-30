"""
Report Generation
=================
Statistical analysis, CSV export, and HTML convergence plots for the
WPT / PSO / GA comparison experiment.

This module is intentionally independent of `run_comparison.py` so it can
be imported and reused (e.g., from a Jupyter notebook) to re-render the
reports from saved raw_results.csv data without re-running the experiment.

Public API
----------
    AlgoRunResult            - dataclass for a single algorithm run
    analyse(results, ecfg)   - print descriptive stats + Friedman / Wilcoxon
    export_csv(results, dir) - write raw_results.csv + convergence_*.csv
    generate_convergence_html(results, dir) - one HTML chart per level
    format_eta(seconds)      - "1h 23min" style ETA formatting
    progress_bar(cur, total) - text progress bar

The convergence outputs use **number of evaluations** as the x-axis
(not iteration index). This matters because WPT performs significantly
more objective evaluations per iteration than PSO/GA, so plotting against
iteration index is misleading. See `analyse_evaluation_budget()` for the
fairness diagnostic.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import stats


# ======================================================================
# Result container
# ======================================================================
@dataclass
class AlgoRunResult:
    """One run of one algorithm on one instance."""
    algorithm: str
    level: str
    instance: int
    run: int
    best_fitness: float
    elapsed_sec: float
    n_evals: int
    convergence: list[float]


# ======================================================================
# Pretty-printing helpers (also used by the orchestrator)
# ======================================================================
def format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}min"


def progress_bar(current: int, total: int, width: int = 30) -> str:
    """Return a text-based progress bar like [████████░░░░░░░] 53%."""
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {frac * 100:5.1f}%"


# ======================================================================
# Statistical analysis
# ======================================================================
def analyse(results: list[AlgoRunResult], ecfg) -> tuple[list[dict], list[dict]]:
    """
    Perform statistical analysis and print report.

    Tests used (following Zitouni et al., 2024):
        1. Descriptive statistics (AVG, STD, MIN, MAX over runs)
        2. Friedman test - non-parametric repeated-measures test across
           all three algorithms per scenario
        3. Wilcoxon signed-rank test - pairwise: WPT vs PSO, WPT vs GA
        4. Mean rank table (as in Table 2 of the paper)
        5. Computation time comparison
        6. Evaluation budget diagnostic (sanity-check fairness)
    """
    algorithms = ["WPT", "PSO", "GA"]
    alpha = 0.05

    print("\n")
    print("=" * 80)
    print("  STATISTICAL ANALYSIS")
    print("=" * 80)

    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    time_grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    eval_grouped: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in results:
        grouped[(r.level, r.instance)][r.algorithm].append(r.best_fitness)
        time_grouped[(r.level, r.instance)][r.algorithm].append(r.elapsed_sec)
        eval_grouped[(r.level, r.instance)][r.algorithm].append(r.n_evals)

    # ------------------------------------------------------------------
    # 1. Descriptive statistics
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  1. DESCRIPTIVE STATISTICS (best fitness over runs)")
    print("-" * 80)
    print(f"  {'Level':<15} {'Inst':>4}  ", end="")
    for alg in algorithms:
        print(f"  {'AVG('+alg+')':>14} {'STD('+alg+')':>14}", end="")
    print()

    desc_rows = []
    for (level, inst), alg_dict in sorted(grouped.items()):
        row = {"level": level, "instance": inst}
        print(f"  {level:<15} {inst:>4}  ", end="")
        for alg in algorithms:
            vals = np.array(alg_dict[alg])
            avg, std = vals.mean(), vals.std()
            row[f"{alg}_avg"] = avg
            row[f"{alg}_std"] = std
            row[f"{alg}_min"] = vals.min()
            row[f"{alg}_max"] = vals.max()
            print(f"  {avg:>14,.2f} {std:>14,.2f}", end="")
        print()
        desc_rows.append(row)

    # ------------------------------------------------------------------
    # 2. Friedman test per scenario
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  2. FRIEDMAN TEST (H0: all algorithms perform equally)")
    print(f"     Significance level alpha = {alpha}")
    print("-" * 80)
    print(f"  {'Level':<15} {'Inst':>4}  {'chi2':>10}  {'p-value':>12}  {'Significant?':>13}")

    friedman_results = []
    for (level, inst), alg_dict in sorted(grouped.items()):
        samples = [np.array(alg_dict[alg]) for alg in algorithms]
        min_len = min(len(s) for s in samples)
        samples = [s[:min_len] for s in samples]

        try:
            stat, pval = stats.friedmanchisquare(*samples)
            sig = "YES" if pval < alpha else "no"
            print(f"  {level:<15} {inst:>4}  {stat:>10.4f}  {pval:>12.2e}  {sig:>13}")
            friedman_results.append({
                "level": level, "instance": inst,
                "chi2": stat, "p_value": pval, "significant": pval < alpha,
            })
        except ValueError as e:
            print(f"  {level:<15} {inst:>4}  skipped ({e})")

    # ------------------------------------------------------------------
    # 3. Wilcoxon signed-rank tests (pairwise)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  3. WILCOXON SIGNED-RANK TEST (pairwise)")
    print(f"     Significance level alpha = {alpha}")
    print("-" * 80)

    pairs = [("WPT", "PSO"), ("WPT", "GA")]
    for a, b in pairs:
        print(f"\n  {a} vs {b}:")
        print(f"  {'Level':<15} {'Inst':>4}  {'W-stat':>10}  {'p-value':>12}  "
              f"{'Winner':>8}  {'Significant?':>13}")

        for (level, inst), alg_dict in sorted(grouped.items()):
            va = np.array(alg_dict[a])
            vb = np.array(alg_dict[b])
            min_len = min(len(va), len(vb))
            va, vb = va[:min_len], vb[:min_len]

            if np.allclose(va, vb):
                print(f"  {level:<15} {inst:>4}  {'--':>10}  {'--':>12}  {'tie':>8}  {'--':>13}")
                continue

            try:
                stat, pval = stats.wilcoxon(va, vb, alternative='two-sided')
                if pval < alpha:
                    winner = a if va.mean() < vb.mean() else b
                    sig = "YES"
                else:
                    winner = "tie"
                    sig = "no"
                print(f"  {level:<15} {inst:>4}  {stat:>10.2f}  {pval:>12.2e}  "
                      f"{winner:>8}  {sig:>13}")
            except ValueError as e:
                print(f"  {level:<15} {inst:>4}  skipped ({e})")

    # ------------------------------------------------------------------
    # 4. Mean rank table (Friedman ranks, as in Table 2 of the paper)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  4. MEAN RANKS (lower is better)")
    print("-" * 80)

    level_ranks: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {a: [] for a in algorithms}
    )

    for (level, inst), alg_dict in sorted(grouped.items()):
        means = {alg: np.mean(alg_dict[alg]) for alg in algorithms}
        sorted_algs = sorted(means, key=means.get)
        for rank, alg in enumerate(sorted_algs, start=1):
            level_ranks[level][alg].append(rank)

    print(f"  {'Level':<15}", end="")
    for alg in algorithms:
        print(f"  {alg:>10}", end="")
    print()

    for level in ecfg.levels:
        if level not in level_ranks:
            continue
        print(f"  {level:<15}", end="")
        for alg in algorithms:
            ranks = level_ranks[level][alg]
            mean_rank = np.mean(ranks) if ranks else float('nan')
            print(f"  {mean_rank:>10.2f}", end="")
        print()

    # ------------------------------------------------------------------
    # 5. Computation time
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  5. COMPUTATION TIME (seconds, averaged over all runs)")
    print("-" * 80)
    print(f"  {'Level':<15} {'Inst':>4}", end="")
    for alg in algorithms:
        print(f"  {'AVG('+alg+')':>12}", end="")
    print()

    for (level, inst), alg_dict in sorted(time_grouped.items()):
        print(f"  {level:<15} {inst:>4}", end="")
        for alg in algorithms:
            vals = np.array(alg_dict[alg])
            print(f"  {vals.mean():>12.3f}", end="")
        print()

    # ------------------------------------------------------------------
    # 6. Evaluation budget diagnostic (fairness sanity check)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  6. EVALUATION BUDGET (#objective evaluations per run)")
    print("     Fair comparison requires similar evaluation counts across algorithms.")
    print("-" * 80)
    print(f"  {'Level':<15} {'Inst':>4}", end="")
    for alg in algorithms:
        print(f"  {'AVG('+alg+')':>14}", end="")
    print(f"  {'WPT/PSO ratio':>14}")

    for (level, inst), alg_dict in sorted(eval_grouped.items()):
        print(f"  {level:<15} {inst:>4}", end="")
        avgs = {}
        for alg in algorithms:
            vals = np.array(alg_dict[alg])
            avgs[alg] = vals.mean()
            print(f"  {avgs[alg]:>14,.0f}", end="")
        ratio = avgs["WPT"] / avgs["PSO"] if avgs["PSO"] > 0 else float('nan')
        print(f"  {ratio:>14.2f}")

    return desc_rows, friedman_results


# ======================================================================
# CSV export
# ======================================================================
def export_csv(results: list[AlgoRunResult], output_dir: str) -> None:
    """Write raw results and per-level convergence-vs-evaluations CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    # Raw results
    path = os.path.join(output_dir, "raw_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "level", "instance", "run",
            "best_fitness", "elapsed_sec", "n_evals",
        ])
        for r in results:
            writer.writerow([
                r.algorithm, r.level, r.instance, r.run,
                r.best_fitness, r.elapsed_sec, r.n_evals,
            ])
    print(f"\n  Raw results saved to: {path}")

    # Convergence by evaluations (one file per level)
    levels = sorted(set(r.level for r in results))
    for level in levels:
        path = os.path.join(output_dir, f"convergence_{level}.csv")
        level_results = [r for r in results if r.level == level]

        sample_results = []
        for alg in ["WPT", "PSO", "GA"]:
            sample = next((r for r in level_results
                           if r.algorithm == alg
                           and r.instance == 0 and r.run == 0), None)
            if sample is not None:
                sample_results.append(sample)

        if not sample_results:
            continue

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["evaluations"] + [s.algorithm for s in sample_results]
            )
            # Use a common evaluation grid: 200 points up to max(n_evals)
            max_evals = max(s.n_evals for s in sample_results)
            grid = np.linspace(0, max_evals, 201)[1:].astype(int)

            for ev in grid:
                row = [int(ev)]
                for s in sample_results:
                    # Map evaluation count to iteration index
                    iters = len(s.convergence)
                    # Iter at which we've spent `ev` evaluations
                    frac = min(ev / max(s.n_evals, 1), 1.0)
                    idx = min(int(frac * iters), iters - 1)
                    row.append(s.convergence[idx] if iters > 0 else "")
                writer.writerow(row)
        print(f"  Convergence (by evaluations) saved to: {path}")


# ======================================================================
# HTML convergence plots
# ======================================================================
def generate_convergence_html(
    results: list[AlgoRunResult], output_dir: str
) -> None:
    """Create a Chart.js HTML page per level with evaluations on the x-axis."""
    os.makedirs(output_dir, exist_ok=True)
    levels = sorted(set(r.level for r in results))

    for level in levels:
        level_results = [r for r in results if r.level == level]
        curves: dict[str, list[list[float]]] = {}
        for alg in ["WPT", "PSO", "GA"]:
            sample = next((r for r in level_results
                           if r.algorithm == alg
                           and r.instance == 0 and r.run == 0), None)
            if sample is None or not sample.convergence:
                continue
            iters = len(sample.convergence)
            # Build (evaluations, fitness) pairs
            # Approximate evals at iteration i: (i+1) * (n_evals / iters)
            evals_per_iter = sample.n_evals / iters
            xy = [
                [int(round((i + 1) * evals_per_iter)), float(v)]
                for i, v in enumerate(sample.convergence)
            ]
            curves[alg] = xy

        if not curves:
            continue

        path = os.path.join(output_dir, f"convergence_{level}.html")
        data_json = json.dumps(curves)

        html = f"""<!DOCTYPE html>
<html><head><title>Convergence - {level}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head><body>
<h2>Convergence Curves - {level.upper()} scenario (x-axis: # objective evaluations)</h2>
<canvas id="chart" width="900" height="400"></canvas>
<script>
const data = {data_json};
const colors = {{"WPT": "#e74c3c", "PSO": "#3498db", "GA": "#2ecc71"}};
const datasets = Object.entries(data).map(([name, xy]) => ({{
    label: name,
    data: xy.map(([x, y]) => ({{x: x, y: y}})),
    borderColor: colors[name],
    fill: false,
    pointRadius: 0,
    borderWidth: 2,
}}));
new Chart(document.getElementById('chart'), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
        parsing: false,
        scales: {{
            x: {{ type: 'linear', title: {{ display: true, text: 'Objective evaluations' }} }},
            y: {{ title: {{ display: true, text: 'Best Cost' }}, type: 'logarithmic' }}
        }},
        plugins: {{ title: {{ display: true,
                              text: 'Convergence: {level.upper()} (vs. evaluation budget)' }} }}
    }}
}});
</script></body></html>"""
        with open(path, "w") as f:
            f.write(html)
        print(f"  Convergence plot saved to: {path}")
