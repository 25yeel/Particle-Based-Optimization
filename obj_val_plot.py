import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bc_algorithms import ackley, cbo, naive
from variant_1 import component_wise_diff
from variant_2 import common_noise
from variant_3 import random_mini_batch


# best f(x) found SO FAR over iterations
def best_obj_val_hist(ax, best_val_hist, best_val):
    # plot x (iteration) and y (obj value)
    ax.semilogy(best_val_hist, color="#1D9E75", linewidth=1.4)
    # draw horizontal y line to show where run ended up
    ax.axhline(y=best_val, color="#D85A30", linestyle="--", linewidth=1, label=(f"Final best: {best_val:.4f}"))

    # ax.set_title("Best f(x) over iterations")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best f(x) found SO FAR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

# best f(x) during each iteration
def obj_val_hist(ax, obj_val_hist):
    # plot x (iteration) and y (obj value)
    ax.semilogy(obj_val_hist, color="blue", linewidth=1.4)
    # draw horizontal y line to show where run ended up
    ax.axhline(y=obj_val_hist[-1], color="#D83030", linestyle="--", linewidth=1, label=(f"Final best: {obj_val_hist[-1]:.4f}"))

    ax.set_xlabel("Iteration"); ax.set_ylabel("Best f(x) found")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

# consensus point history over iterations
def consensus_hist(ax, consensus_hist):
    consensus_vals = ackley(consensus_hist)
    ax.semilogy(consensus_vals, color="#06BF15", linewidth=1.4)
    # draw horizontal y line to show where run ended up
    ax.axhline(y=consensus_vals[-1], color="#591766", linestyle="--", linewidth=1, label=(f"Final best: {consensus_vals[-1]:.4f}"))

    ax.set_xlabel("Iteration"); ax.set_ylabel("Consensus point found")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

def obj_val_naive(ax):
    best_x, best_val, x_positions, f_vals, n_particles, variance = naive(ackley)
    # draw horizontal line to compare best_val to other graphs
    ax.axhline(y=best_val, color="#1D9E75", linestyle="-", linewidth=1.4, label=(f"Best f val: {best_val:.4f}"))
    ax.set_title("NAIVE ALGORITHM Best f(x)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best f(x) found")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def main():
    # create blank canvas
    fig = plt.figure(figsize=(15,12))
    fig.suptitle("CBO on the 2D Ackley Function", fontsize=14, fontweight="bold")

    # 2x2 figure panels
    ax1 = fig.add_subplot(5,3,1)
    ax2 = fig.add_subplot(5,3,2)
    ax3 = fig.add_subplot(5,3,3)
    ax4 = fig.add_subplot(5,3,4)
    ax5 = fig.add_subplot(5,3,5)
    ax6 = fig.add_subplot(5,3,6)
    ax7 = fig.add_subplot(5,3,7)
    ax8 = fig.add_subplot(5,3,8)
    ax9 = fig.add_subplot(5,3,9)
    ax10 = fig.add_subplot(5,3,10)
    ax11 = fig.add_subplot(5,3,11)
    ax12 = fig.add_subplot(5,3,12)
    ax13 = fig.add_subplot(5,3,13)
    # ax14 = fig.add_subplot(5,3,14)
    # ax15 = fig.add_subplot(5,3,15)

    _, cbo_best_val, cbo_consensus_hist, _, cbo_obj_val_hist, cbo_best_val_hist, _ = cbo(ackley, d=2, seed=0)
    _, var1_best_val, var1_consensus_hist, _, var1_obj_val_hist, var1_best_val_hist, _ = component_wise_diff(ackley, d=2, seed=0)
    _, var2_best_val, var2_consensus_hist, _, var2_obj_val_hist, var2_best_val_hist, _ = common_noise(ackley, d=2, seed=0)
    _, var3_best_val, var3_consensus_hist, _, var3_obj_val_hist, var3_best_val_hist, _ = random_mini_batch(ackley, d=2, seed=0)

    # plot graphs
    # running best obj val history
    best_obj_val_hist(ax1, cbo_best_val_hist, cbo_best_val)
    ax1.set_title("BC CBO Running Best f(x) over iterations")
    best_obj_val_hist(ax4, var1_best_val_hist, var1_best_val)
    ax4.set_title("VARIANT 1 Running Best f(x) over iterations")
    best_obj_val_hist(ax7, var2_best_val_hist, var2_best_val)
    ax7.set_title("VARIANT 2 Running Best f(x) over iterations")
    best_obj_val_hist(ax10, var3_best_val_hist, var3_best_val)
    ax10.set_title("VARIANT 3 Running Best f(x) over iterations")
    obj_val_naive(ax13)

    # obj value history
    obj_val_hist(ax2, cbo_obj_val_hist)
    ax2.set_title("BC CBO Best f(x) over iterations")
    obj_val_hist(ax5, var1_obj_val_hist)
    ax5.set_title("VARIANT 1 Best f(x) over iterations")
    obj_val_hist(ax8, var2_obj_val_hist)
    ax8.set_title("VARIANT 2 Best f(x) over iterations")
    obj_val_hist(ax11, var3_obj_val_hist)
    ax11.set_title("VARIANT 3 Best f(x) over iterations")

    # consensus point history
    consensus_hist(ax3, cbo_consensus_hist)
    ax3.set_title("BC CBO Consensus point over iterations")
    consensus_hist(ax6, var1_consensus_hist)
    ax6.set_title("VARIANT 1 Consensus point over iterations")
    consensus_hist(ax9, var2_consensus_hist)
    ax9.set_title("VARIANT 2 Consensus point over iterations")
    consensus_hist(ax12, var3_consensus_hist)
    ax12.set_title("VARIANT 3 Consensus point over iterations")


    plt.tight_layout()
    plt.savefig("obj_val_plot.png", dpi=150)

if __name__ == "__main__":
    main()
