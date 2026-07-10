import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bc_algorithms import ackley, cbo
from variant_1 import component_wise_diff
from variant_2 import common_noise


def grid(res):
    # create grid
    lin = np.linspace(-32, 32, res)
    X, Y = np.meshgrid(lin, lin)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

    # evaluate function at all grid points, correspond to grid coord
    Z = ackley(grid).reshape(res, res)

    return X, Y, Z

# VARIANCE PLOT
def variance_hist(ax, var_hist, n_iters):
    # plot x (iteration) and y (variance of particle distr)
    ax.semilogy(var_hist, color="#5A6DD8", linewidth=1.8)
    # draw vertical x line at index n_iters-1 to mark when the stopping criterion was met
    ax.axvline(x=n_iters-1, color="#CF179B", linestyle=":", linewidth=1, label=(f"Stopped at iter {n_iters}"))

    ax.set_title("Loss: Variance V(t) over iterations")
    ax.set_xlabel("Iteration"); ax.set_ylabel("V(t)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

def main():
    # create blank canvas
    fig = plt.figure(figsize=(13,10))
    fig.suptitle("CBO on the 2D Ackley Function", fontsize=14, fontweight="bold")

    # 2x2 figure panels
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    # ax4 = fig.add_subplot(2,2,4)

    # plot graphs
    _, _, _, _, _, _, cbo_var_hist = cbo(ackley, d=2, seed=0)
    _, _, _, _, _, _, var1_var_hist = component_wise_diff(ackley, d=2, seed=0)
    _, _, _, _, _, _, var2_var_hist = common_noise(ackley, d=2, seed=0)
    variance_hist(ax1, cbo_var_hist, len(cbo_var_hist))
    ax1.set_title("BC CBO Variance")
    variance_hist(ax2, var1_var_hist, len(var1_var_hist))
    ax2.set_title("VARIANT 1 Variance w/ Component-Wise Diffusion")
    variance_hist(ax3, var2_var_hist, len(var2_var_hist))
    ax3.set_title("VARIANT 2 Variance w/ Component-Wise Diffusion & Common Noise")


    plt.tight_layout()
    plt.savefig("variance_plot.png", dpi=150)

if __name__ == "__main__":
    main()
