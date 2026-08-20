# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from bc_algorithms import ackley, cbo, naive, naive_uniform
from variants_cbo.variant_1 import component_wise_diff
from variants_cbo.variant_2 import common_noise
from variants_cbo.variant_3 import random_mini_batch
from variants_cbo.variant_1and3 import comp_diff_random_batch


# best f(x) found SO FAR over iterations
def best_obj_val_hist(ax, best_val_hist, best_val, n_particles=50, m_size=50):
    # plot x (iteration) and y (obj value)
    ax.semilogy(best_val_hist, color="#1D9E75", linewidth=1.4)
    # draw horizontal y line to show where run ended up
    ax.axhline(y=best_val, color="#D85A30", linestyle="--", linewidth=1, label=(f"Final best: {best_val:.4f}"))

    # ax.set_title("Best f(x) over iterations")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best f(x) found SO FAR")
    # ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # get the existing plot handles automatically
    handles, labels = ax.get_legend_handles_labels()

    # add info patches to legend
    info1 = mpatches.Patch(color='none', label=f'NFE: {m_size * len(best_val_hist)}')
    info2 = mpatches.Patch(color='none', label=f'N: {n_particles}')
    info3 = mpatches.Patch(color='none', label=f'M: {m_size}')
    ax.legend(handles=handles + [info1, info2, info3], fontsize=8)

# best f(x) during each iteration
def obj_val_hist(ax, obj_val_hist, n_particles=50, m_size=50):
    # plot x (iteration) and y (obj value)
    ax.semilogy(obj_val_hist, color="blue", linewidth=1.4)
    # draw horizontal y line to show where run ended up
    ax.axhline(y=obj_val_hist[-1], color="#D83030", linestyle="--", linewidth=1, label=(f"Final best: {obj_val_hist[-1]:.4f}"))

    ax.set_xlabel("Iteration"); ax.set_ylabel("Best f(x) found")
    # ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # add info patches to legend
    handles, labels = ax.get_legend_handles_labels()
    info1 = mpatches.Patch(color='none', label=f'NFE: {m_size*len(obj_val_hist)}')
    info2 = mpatches.Patch(color='none', label=f'N: {n_particles}')
    info3 = mpatches.Patch(color='none', label=f'M: {m_size}')
    ax.legend(handles=handles + [info1, info2, info3], fontsize=8)

# consensus point history over iterations
def consensus_hist(ax, consensus_hist, n_particles=50, m_size=50):
    consensus_vals = ackley(consensus_hist)
    ax.semilogy(consensus_vals, color="#06BF15", linewidth=1.4)
    # draw horizontal y line to show where run ended up
    ax.axhline(y=consensus_vals[-1], color="#591766", linestyle="--", linewidth=1, label=(f"Final best: {consensus_vals[-1]:.4f}"))

    ax.set_xlabel("Iteration"); ax.set_ylabel("Consensus point found")
    # ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # add info patches to legend
    handles, labels = ax.get_legend_handles_labels()
    info1 = mpatches.Patch(color='none', label=f'NFE: {m_size*len(consensus_hist)}')
    info2 = mpatches.Patch(color='none', label=f'N: {n_particles}')
    info3 = mpatches.Patch(color='none', label=f'M: {m_size}')
    ax.legend(handles=handles + [info1, info2, info3], fontsize=8)

def obj_val_naive(ax, f=naive):
    best_x, best_val, x_positions, f_vals, n_particles, variance = f(ackley, d=2)
    # draw horizontal line to compare best_val to other graphs
    ax.axhline(y=best_val, color="#1D9E75", linestyle="-", linewidth=1.4, label=(f"Best f val: {best_val:.4f}"))
    ax.set_title("NAIVE ALGORITHM Best f(x)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best f(x) found")
    # ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # add info patches to legend
    handles, labels = ax.get_legend_handles_labels()
    info1 = mpatches.Patch(color='none', label=f'NFE: {n_particles}')
    info2 = mpatches.Patch(color='none', label=f'N: {n_particles}')
    ax.legend(handles=handles + [info1, info2], fontsize=8)


def main():
    # create blank canvas
    fig = plt.figure(figsize=(15,15))
    fig.suptitle("CBO on the 2D Ackley Function", fontsize=14, fontweight="bold")

    # 2x2 figure panels
    ax1 = fig.add_subplot(6,3,1)
    ax2 = fig.add_subplot(6,3,2)
    ax3 = fig.add_subplot(6,3,3)
    ax4 = fig.add_subplot(6,3,4)
    ax5 = fig.add_subplot(6,3,5)
    ax6 = fig.add_subplot(6,3,6)
    ax7 = fig.add_subplot(6,3,7)
    ax8 = fig.add_subplot(6,3,8)
    ax9 = fig.add_subplot(6,3,9)
    ax10 = fig.add_subplot(6,3,10)
    ax11 = fig.add_subplot(6,3,11)
    ax12 = fig.add_subplot(6,3,12)
    ax13 = fig.add_subplot(6,3,13)
    ax14 = fig.add_subplot(6,3,14)
    ax15 = fig.add_subplot(6,3,15)
    ax16 = fig.add_subplot(6,3,16)
    ax17 = fig.add_subplot(6,3,17)


    _, cbo_best_val, cbo_consensus_hist, _, cbo_obj_val_hist, cbo_best_val_hist, _ = cbo(ackley, d=2, seed=0)
    _, var1_best_val, var1_consensus_hist, _, var1_obj_val_hist, var1_best_val_hist, _ = component_wise_diff(ackley, d=2, seed=0)
    _, var2_best_val, var2_consensus_hist, _, var2_obj_val_hist, var2_best_val_hist, _ = common_noise(ackley, d=2, seed=0)
    _, var3_best_val, var3_consensus_hist, _, var3_obj_val_hist, var3_best_val_hist, _ = random_mini_batch(ackley, d=2, seed=0)
    _, var13_best_val, var13_consensus_hist, _, var13_obj_val_hist, var13_best_val_hist, _ = comp_diff_random_batch(ackley, d=2, seed=0)

    # plot graphs
    # running best obj val history
    best_obj_val_hist(ax1, cbo_best_val_hist, cbo_best_val)
    ax1.set_title("BC CBO Running Best f(x) over iterations")
    best_obj_val_hist(ax4, var1_best_val_hist, var1_best_val)
    ax4.set_title("VARIANT 1 Running Best f(x) over iterations")
    best_obj_val_hist(ax7, var2_best_val_hist, var2_best_val)
    ax7.set_title("VARIANT 2 Running Best f(x) over iterations")
    best_obj_val_hist(ax10, var3_best_val_hist, var3_best_val, m_size=10)
    ax10.set_title("VARIANT 3 Running Best f(x) over iterations")
    best_obj_val_hist(ax13, var13_best_val_hist, var13_best_val, m_size=10)
    ax13.set_title("VARIANT 1+3 Running Best f(x) over iterations")
    obj_val_naive(ax16)
    obj_val_naive(ax17, f=naive_uniform)
    ax17.set_title("NAIVE ALGORITHM (UNIFORM GRID DISTR) Best f(x)")


    # obj value history
    obj_val_hist(ax2, cbo_obj_val_hist)
    ax2.set_title("BC CBO Best f(x) over iterations")
    obj_val_hist(ax5, var1_obj_val_hist)
    ax5.set_title("VARIANT 1 Best f(x) over iterations")
    obj_val_hist(ax8, var2_obj_val_hist)
    ax8.set_title("VARIANT 2 Best f(x) over iterations")
    obj_val_hist(ax11, var3_obj_val_hist) # obj_val_hist stays full-swarm-sourced (n_particles cost), unlike best_val_hist
    ax11.set_title("VARIANT 3 Best f(x) over iterations")
    obj_val_hist(ax14, var13_obj_val_hist) # same: full-swarm-sourced, not mini-batch
    ax14.set_title("VARIANT 1+3 Best f(x) over iterations")

    # consensus point history
    consensus_hist(ax3, cbo_consensus_hist)
    ax3.set_title("BC CBO Consensus point over iterations")
    consensus_hist(ax6, var1_consensus_hist)
    ax6.set_title("VARIANT 1 Consensus point over iterations")
    consensus_hist(ax9, var2_consensus_hist)
    ax9.set_title("VARIANT 2 Consensus point over iterations")
    consensus_hist(ax12, var3_consensus_hist, m_size=10)
    ax12.set_title("VARIANT 3 Consensus point over iterations")
    consensus_hist(ax15, var13_consensus_hist, m_size=10)
    ax15.set_title("VARIANT 1+3 Consensus point over iterations")


    plt.tight_layout()
    plt.savefig("obj_val_plot.png", dpi=150)

if __name__ == "__main__":
    main()
