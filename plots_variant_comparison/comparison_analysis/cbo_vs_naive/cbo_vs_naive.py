# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import csv
import numpy as np
import matplotlib.pyplot as plt

from bc_algorithms import ackley, rastrigin, cbo, naive, naive_uniform

def plot(f, d=2, seed=0):

    """HYPERPARAMS ARE THE SAME AS IN bc_algorithms.py"""
    best_x, running_best, consensus_hist, part_hist, obj_val_hist, best_val_hist, var_hist = cbo(f, d=d, seed=seed)
    n_particles = part_hist.shape[1]
    n_iters = len(best_val_hist)
    NFE_cbo = np.arange(1, n_iters+1) * n_particles # each iter costs n_particles evaluations

    best_x_naive, best_val_naive, x_positions_naive, f_vals_naive, n_evals_naive, var_naive = naive(f, d=d, seed=seed)
    # naive has no iterations (draws and evaluates one big independent and identically distributed batch at once)
    # only output of naive is (NFE_total, best_val) point
    # so plot at every possible NFE

    naive_running_best = np.minimum.accumulate(f_vals_naive)
    NFE_naive = np.arange(1, len(f_vals_naive)+1)

    # SWARM: every individual evaluation, not just running best
    # use part_hist to get objective values for every particle
    cbo_swarm_vals = f(part_hist.reshape(-1, d)).reshape(n_iters, n_particles)
    cbo_swarm_NFE = np.repeat(NFE_cbo, n_particles)
    # for naive swarm, already have all points
    naive_swarm_vals = f_vals_naive
    naive_swarm_NFE = NFE_naive

    # summary table
    headers = ["Algorithm", "Best Obj Value", "Total NFE", "Final Variance", "Iterations/Samples"]
    rows = [
        ["CBO",   f"{best_val_hist[-1]:.6f}", f"{NFE_cbo[-1]}",   f"{var_hist[-1]:.6f}", f"{n_iters}"],
        ["Naive", f"{best_val_naive:.6f}",    f"{NFE_naive[-1]}", f"{var_naive:.6f}",    f"{n_evals_naive}"],
    ]
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(vals):
        return " | ".join(v.ljust(w) for v, w in zip(vals, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))

    with open("cbo_vs_naive.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)

    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,10), sharex=True)
    # curve
    ax1.plot(NFE_cbo, best_val_hist, label="CBO")
    ax1.plot(NFE_naive, naive_running_best, label="Naive (random search)")
    # ax1.set_yscale("log")
    ax1.set_ylabel("Best objective value found so far")
    ax1.set_title("Best Value vs. NFE")
    ax1.legend()
    ax1.grid(alpha=0.3)
    # scatter
    ax2.scatter(cbo_swarm_NFE, cbo_swarm_vals.ravel(), s=8, alpha=0.3, label="CBO (all particles)")
    ax2.scatter(naive_swarm_NFE, naive_swarm_vals, s=4, alpha=0.15, label="Naive (all samples)")
    # ax2.set_yscale("log")
    ax2.set_xlabel("NFE"); ax2.set_ylabel("Objective value (every particle evaluation)")
    ax2.set_title("Full Swarm/Sample Spread vs. NFE")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("CBO vs. Naive Search", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cbo_vs_naive.png", dpi=150)

def main():
    plot(ackley)

if __name__ == "__main__":
    main()
