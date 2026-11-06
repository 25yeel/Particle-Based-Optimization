# LAST UPDATED 11/6/2026
# Base CBO vs. ZO-AdaMM (batch of particles), compared on NFE (number of function evaluations)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import csv
import numpy as np
import matplotlib.pyplot as plt

from bc_algorithms import ackley, rastrigin, cbo, zo_adamm_batch

OUT_DIR = Path(__file__).resolve().parent


def plot(f, name, d=2, domain=(-32, 32), seed=0, zo_particles=25, q=10, zo_iters=1000):

    """HYPERPARAMS ARE THE SAME AS IN bc_algorithms.py"""
    # BC CBO
    best_x, running_best, consensus_hist, part_hist, obj_val_hist, best_val_hist, var_hist = cbo(f, d=d, domain=domain, seed=seed)
    n_particles = part_hist.shape[1]
    n_iters = len(best_val_hist)
    NFE_cbo = np.arange(1, n_iters+1) * n_particles # each iter costs n_particles evaluations

    # ZO-AdaMM (batch of independent particles)
    # each iter costs n_particles * (q+1) evaluations: f(x) once + f(x + mu*u) for each of the q directions
    zo_best_x, zo_running_best, zo_part_hist, zo_f_vals, zo_obj_val_hist, zo_best_val_hist = zo_adamm_batch(
        f, d=d, domain=domain, n_particles=zo_particles, q=q, iters=zo_iters, seed=seed)
    n_iters_zo = len(zo_best_val_hist)
    NFE_zo = np.arange(1, n_iters_zo+1) * zo_particles * (q+1)

    # ZO-AdaMM at the SAME budget CBO used (CBO stops early once its variance < tol)
    budget = NFE_cbo[-1]
    within = NFE_zo <= budget
    zo_at_budget = zo_best_val_hist[within][-1] if within.any() else np.nan
    # NFE ZO-AdaMM needs to match CBO's final best value (nan if it never does)
    hit = np.nonzero(zo_best_val_hist <= best_val_hist[-1])[0]
    zo_NFE_to_match = NFE_zo[hit[0]] if hit.size else np.nan

    # summary table
    headers = ["Algorithm", "Best Obj Value", "Total NFE", f"Best @ CBO budget ({budget})", "NFE to reach CBO best", "Iterations"]
    rows = [
        ["CBO", f"{best_val_hist[-1]:.6f}", f"{NFE_cbo[-1]}", f"{best_val_hist[-1]:.6f}", f"{NFE_cbo[-1]}", f"{n_iters}"],
        [f"ZO-AdaMM (N={zo_particles}, q={q})", f"{zo_running_best:.6f}", f"{NFE_zo[-1]}", f"{zo_at_budget:.6f}", f"{zo_NFE_to_match}", f"{n_iters_zo}"],
    ]
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(vals):
        return " | ".join(v.ljust(w) for v, w in zip(vals, col_widths))

    print(f"\n{name} (d={d}, domain={domain})")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))

    with open(OUT_DIR / f"cbo_vs_zo_adamm_{name}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)

    # plot
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.plot(NFE_cbo, best_val_hist, label="CBO")
    ax1.plot(NFE_zo, zo_best_val_hist, label=f"ZO-AdaMM (N={zo_particles}, q={q})")
    ax1.axvline(budget, color="gray", ls="--", lw=1, label=f"CBO total NFE ({budget})")
    ax1.set_xscale("log"); ax1.set_yscale("log") # ZO-AdaMM uses ~(q+1)x more NFE per iter, log x keeps both readable
    ax1.set_xlabel("NFE"); ax1.set_ylabel("Best objective value found so far")
    ax1.set_title("Best Value vs. NFE")
    ax1.legend()
    ax1.grid(alpha=0.3)

    fig.suptitle(f"CBO vs. ZO-AdaMM ({name.capitalize()}, d={d})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cbo_vs_zo_adamm_{name}.png", dpi=150)
    plt.close(fig)


def main():
    plot(ackley, "ackley", d=2, domain=(-32, 32))
    plot(rastrigin, "rastrigin", d=2, domain=(-5.12, 5.12))

if __name__ == "__main__":
    main()
