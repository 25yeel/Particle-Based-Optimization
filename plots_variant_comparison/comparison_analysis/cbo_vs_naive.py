import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from bc_algorithms import ackley, cbo, naive


def main():
    d = 2
    seed = 0

    # naive() always benchmarks its NFE budget against a FRESH cbo(f, d) call using
    # plain defaults internally -- so this cbo() call also uses defaults, keeping both
    # curves on the same NFE budget for a fair comparison
    best_x, running_best, consensus_hist, particle_hist, obj_val_hist, best_val_hist, variance_hist = cbo(ackley, d=d, seed=seed)

    n_particles = particle_hist.shape[1]
    n_iters = len(best_val_hist)
    NFE_cbo = np.arange(1, n_iters + 1) * n_particles # each iteration costs n_particles evals

    best_x_naive, best_val_naive, x_positions_naive, f_vals_naive, n_evals_naive, variance_naive = naive(ackley, d=d, seed=seed)

    # naive() has no iterations -- it draws and evaluates one big i.i.d. batch at once,
    # so its only TRUE output is a single (NFE_total, best_val) point. The running min
    # below is an interpretive "as-if-truncated" curve: valid since the samples are i.i.d.
    # (so any prefix is itself a smaller-budget random search), but not what naive()
    # actually computes step-by-step -- plotted lightly/dashed to keep that distinction visible
    naive_running_best = np.minimum.accumulate(f_vals_naive)
    NFE_naive = np.arange(1, len(f_vals_naive) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(NFE_cbo, best_val_hist, label="CBO")
    ax.plot(NFE_naive, naive_running_best, linestyle="--", alpha=0.5, color="gray",
            label="Naive (as-if-truncated, interpretive)")
    ax.scatter([NFE_naive[-1]], [best_val_naive], color="tab:orange", zorder=5,
               label="Naive (true output: full-budget result)")

    ax.set_xlabel("NFE (number of function evaluations)")
    ax.set_ylabel("Best objective value found so far")
    ax.set_yscale("log")
    ax.set_title("CBO vs. Naive Search: Objective Value vs. NFE")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("cbo_vs_naive.png", dpi=150)


if __name__ == "__main__":
    main()
