# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import csv
import numpy as np
import matplotlib.pyplot as plt

from bc_algorithms import ackley, rastrigin, cbo, naive, naive_uniform
from variants_cbo.variant_1 import component_wise_diff
from variants_cbo.variant_2 import common_noise
from variants_cbo.variant_3 import random_mini_batch
from variants_cbo.variant_1and3 import comp_diff_random_batch


def plot(f, d=2, seed=0):

    """HYPERPARAMS ARE THE SAME AS IN bc_algorithms.py"""
    # BC CBO
    best_x, running_best, consensus_hist, part_hist, obj_val_hist, best_val_hist, var_hist = cbo(f, d=d, seed=seed)
    n_particles = part_hist.shape[1]
    n_iters = len(best_val_hist)
    NFE_cbo = np.arange(1, n_iters+1) * n_particles # each iter costs n_particles evaluations

    # variant 1
    *_, var1_part_hist, _, var1_best_val_hist, var1_var_hist = component_wise_diff(f, d=d, seed=seed)
    n_particles_var1 = part_hist.shape[1]
    n_iters_var1 = len(var1_best_val_hist)
    NFE_var1 = np.arange(1, n_iters_var1+1) * n_particles_var1

    # variant 2
    *_, var2_part_hist, _, var2_best_val_hist, var2_var_hist = common_noise(f, d=d, seed=seed)
    n_particles_var2 = part_hist.shape[1]
    n_iters_var2 = len(var2_best_val_hist)
    NFE_var2 = np.arange(1, n_iters_var2+1) * n_particles_var2

    # variant 3
    # batch_size must match random_mini_batch's batch_size param (default 10) --
    # best_val_hist is sourced from the mini-batch, so NFE = batch_size*n_iters, not n_particles*n_iters
    batch_size_var3 = 10
    *_, var3_part_hist, _, var3_best_val_hist, var3_var_hist = random_mini_batch(f, d=d, seed=seed, batch_size=batch_size_var3)
    n_iters_var3 = len(var3_best_val_hist)
    NFE_var3 = np.arange(1, n_iters_var3+1) * batch_size_var3

    # variant 1&3
    # same mini-batch NFE accounting as variant 3, per comp_diff_random_batch's batch_size param
    batch_size_var1and3 = 10
    *_, var1and3_part_hist, _, var1and3_best_val_hist, var1and3_var_hist = comp_diff_random_batch(f, d=d, seed=seed, batch_size=batch_size_var1and3)
    n_iters_var1and3 = len(var1and3_best_val_hist)
    NFE_var1and3 = np.arange(1, n_iters_var1and3+1) * batch_size_var1and3

    """best_x_naive, best_val_naive, x_positions_naive, f_vals_naive, n_evals_naive, var_naive = naive(f, d=d, seed=seed)"""
    # naive has no iterations (draws and evaluates one big independent and identically distributed batch at once)
    # only output of naive is (NFE_total, best_val) point
    # so plot at every possible NFE

    # naive_running_best = np.minimum.accumulate(f_vals_naive)
    # NFE_naive = np.arange(1, len(f_vals_naive)+1)

    # SWARM: every individual evaluation, not just running best
    # use part_hist to get objective values for every particle
    """cbo_swarm_vals = f(part_hist.reshape(-1, d)).reshape(n_iters, n_particles)
    cbo_swarm_NFE = np.repeat(NFE_cbo, n_particles)"""
    # for naive swarm, already have all points
    """naive_swarm_vals = f_vals_naive
    naive_swarm_NFE = NFE_naive"""

    # summary table
    headers = ["Algorithm", "Best Obj Value", "Total NFE", "Final Variance", "Iterations/Samples"]
    rows = [
        ["CBO",   f"{best_val_hist[-1]:.6f}", f"{NFE_cbo[-1]}",   f"{var_hist[-1]:.6f}", f"{n_iters}"],
        # ["Naive", f"{best_val_naive:.6f}",    f"{NFE_naive[-1]}", f"{var_naive:.6f}",    f"{n_evals_naive}"],
        ["Var 1",   f"{var1_best_val_hist[-1]:.6f}", f"{NFE_var1[-1]}",   f"{var1_var_hist[-1]:.6f}", f"{n_iters_var1}"],
        ["Var 2",   f"{var2_best_val_hist[-1]:.6f}", f"{NFE_var2[-1]}",   f"{var2_var_hist[-1]:.6f}", f"{n_iters_var2}"],
        ["Var 3",   f"{var3_best_val_hist[-1]:.6f}", f"{NFE_var3[-1]}",   f"{var3_var_hist[-1]:.6f}", f"{n_iters_var3}"],
        ["Var 1 and 3",   f"{var1and3_best_val_hist[-1]:.6f}", f"{NFE_var1and3[-1]}",   f"{var1and3_var_hist[-1]:.6f}", f"{n_iters_var1and3}"],
    ]
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(vals):
        return " | ".join(v.ljust(w) for v, w in zip(vals, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))

    with open("cbo_vs_variants.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)

    # plot
    fig, (ax1 ) = plt.subplots(1, 1, figsize=(10,8), sharex=True)
    # curve
    ax1.plot(NFE_cbo, best_val_hist, label="CBO")
    ax1.plot(NFE_var1, var1_best_val_hist, label="Var 1")
    ax1.plot(NFE_var2, var2_best_val_hist, label="Var 2")
    ax1.plot(NFE_var3, var3_best_val_hist, label="Var 3")
    ax1.plot(NFE_var1and3, var1and3_best_val_hist, label="Var 1 and 3")
    # ax1.plot(NFE_naive, naive_running_best, label="Naive (random search)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Best objective value found so far")
    ax1.set_title("Best Value vs. NFE")
    ax1.legend()
    ax1.grid(alpha=0.3)
    # scatter
    """ax2.scatter(cbo_swarm_NFE, cbo_swarm_vals.ravel(), s=8, alpha=0.3, label="CBO (all particles)")
    ax2.scatter(naive_swarm_NFE, naive_swarm_vals, s=4, alpha=0.15, label="Naive (all samples)")
    # ax2.set_yscale("log")
    ax2.set_xlabel("NFE"); ax2.set_ylabel("Objective value (every particle evaluation)")
    ax2.set_title("Full Swarm/Sample Spread vs. NFE")
    ax2.legend()
    ax2.grid(alpha=0.3)"""

    fig.suptitle("CBO vs. Variants", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cbo_vs_variants.png", dpi=150)

def main():
    plot(ackley)

if __name__ == "__main__":
    main()
