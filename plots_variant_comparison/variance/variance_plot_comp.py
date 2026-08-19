# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from bc_algorithms import ackley
from plots_variant_comparison.var_NFEcomp import naive_fixedNFE, naive_uniform_fixedNFE, cbo_fixedNFE, component_wise_diff_fixedNFE, common_noise_fixedNFE, random_mini_batch_fixedNFE, comp_diff_random_batch_fixedNFE


# VARIANCE PLOT
def variance_hist(ax, var_hist, n_iters, n_particles=50, m_size=50):
    # plot x (iteration) and y (variance of particle distr)
    ax.semilogy(var_hist, color="#5A6DD8", linewidth=1.8)
    # draw vertical x line at index n_iters-1 to mark when the stopping criterion was met
    ax.axvline(x=n_iters-1, color="#CF179B", linestyle=":", linewidth=1, label=(f"Stopped at iter {n_iters}"))

    # ax.set_title("Loss: Variance V(t) over iterations")
    ax.set_xlabel("Iteration"); ax.set_ylabel("V(t)")
    ax.grid(alpha=0.3, which="both")

    # add info patches to legend
    handles, labels = ax.get_legend_handles_labels()
    info1 = mpatches.Patch(color='none', label=f'NFE: {m_size*n_iters}')
    info2 = mpatches.Patch(color='none', label=f'N: {n_particles}')
    info3 = mpatches.Patch(color='none', label=f'M: {m_size}')
    ax.legend(handles=handles + [info1, info2, info3], fontsize=8)

def variance_naive(ax, f=naive_fixedNFE):
    best_x, best_val, x_positions, f_vals, n_particles, variance = f(ackley, d=2)
    # draw horizontal line to compare best_val to other graphs
    ax.axhline(y=variance, color="#5A6DD8", linestyle="-", linewidth=1.8, label=(""))
    ax.set_title("NAIVE ALGORITHM Variance")
    ax.set_xlabel("Iteration"); ax.set_ylabel("V(t)")
    ax.grid(alpha=0.3, which="both")

    # add info patches to legend
    handles, labels = ax.get_legend_handles_labels()
    info1 = mpatches.Patch(color='none', label=f'No iterations')
    info2 = mpatches.Patch(color='none', label=f'NFE: {n_particles}')
    info3 = mpatches.Patch(color='none', label=f'N: {n_particles}')
    ax.legend(handles=handles + [info1, info2, info3], fontsize=8)

def main():
    # create blank canvas
    fig = plt.figure(figsize=(13,10))
    fig.suptitle("CBO on the 2D Ackley Function", fontsize=14, fontweight="bold")

    # 2x2 figure panels
    ax1 = fig.add_subplot(4,2,1)
    ax2 = fig.add_subplot(4,2,2)
    ax3 = fig.add_subplot(4,2,3)
    ax4 = fig.add_subplot(4,2,4)
    ax5 = fig.add_subplot(4,2,5)
    ax6 = fig.add_subplot(4,2,6)
    ax7 = fig.add_subplot(4,2,7)

    # plot graphs
    _, _, _, _, _, _, cbo_var_hist = cbo_fixedNFE(ackley, d=2, seed=0)
    _, _, _, _, _, _, var1_var_hist = component_wise_diff_fixedNFE(ackley, d=2, seed=0)
    _, _, _, _, _, _, var2_var_hist = common_noise_fixedNFE(ackley, d=2, seed=0)
    _, _, _, _, _, _, var3_var_hist = random_mini_batch_fixedNFE(ackley, d=2, seed=0)
    _, _, _, _, _, _, var13_var_hist = comp_diff_random_batch_fixedNFE(ackley, d=2, seed=0)

    variance_hist(ax1, cbo_var_hist, len(cbo_var_hist))
    ax1.set_title("BC CBO Variance")
    variance_hist(ax2, var1_var_hist, len(var1_var_hist))
    ax2.set_title("VARIANT 1 Variance w/ Component-Wise Diffusion")
    variance_hist(ax3, var2_var_hist, len(var2_var_hist))
    ax3.set_title("VARIANT 2 Variance w/ Component-Wise Diffusion & Common Noise")
    variance_hist(ax4, var3_var_hist, len(var3_var_hist), m_size=10)
    ax4.set_title("VARIANT 3 Variance using Random Mini Batches")
    variance_hist(ax5, var13_var_hist, len(var13_var_hist), m_size=10)
    ax5.set_title("VARIANT 1+3 Variance w/ Component-Wise Diffusion using Random Batching")

    variance_naive(ax6)
    variance_naive(ax7, f=naive_uniform_fixedNFE)
    ax7.set_title("NAIVE ALGORITHM (UNIFORM GRID DISTR) Variance")

    plt.tight_layout()
    plt.savefig("variance_plot_comp.png", dpi=150)

if __name__ == "__main__":
    main()
