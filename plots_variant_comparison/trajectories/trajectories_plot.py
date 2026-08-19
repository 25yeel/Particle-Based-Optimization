import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from bc_algorithms import ackley, cbo, naive, naive_uniform
from variants_cbo.variant_1 import component_wise_diff
from variants_cbo.variant_2 import common_noise
from variants_cbo.variant_3 import random_mini_batch
from variants_cbo.variant_1and3 import comp_diff_random_batch


def grid(res):
    # create grid
    lin = np.linspace(-32, 32, res)
    X, Y = np.meshgrid(lin, lin)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

    # evaluate function at all grid points, correspond to grid coord
    Z = ackley(grid).reshape(res, res)

    return X, Y, Z

# TRAJECTORY PLOT
def particle_hist(ax, particle_hist, m_size=50):
    _, _, Z = grid(res=300)

    # draw 2d array heatmap
    ax.imshow(Z, extent=[-32, 32, -32, 32], origin="lower", cmap="Blues", aspect="equal", norm=mcolors.PowerNorm(gamma=0.4,vmin=0,vmax=22))

    n_particles = particle_hist.shape[1]
    n_iters = len(particle_hist)

    for i in range(n_particles):
        traj = particle_hist[:,i,:] # draw one trajectory line per particle
        x_1 = traj[:,0]
        x_2 = traj[:,1]
        ax.plot(x_1, x_2, color="white", alpha=0.15, linewidth=1.0)
    # plot individual points at iteration 0
    ax.scatter(particle_hist[0,:,0], particle_hist[0,:,1], color="orange", s=12, zorder=3, label="Start")
    # plot individual points at last iteration
    ax.scatter(particle_hist[-1,:,0], particle_hist[-1,:,1], color="red", s=12, zorder=3, label="End")

    # styling
    # ax.set_title("Particle Trajectories")
    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
    ax.set_xlim(-32, 32); ax.set_ylim(-32, 32)

    # add info patches to legend
    handles, labels = ax.get_legend_handles_labels()
    info1 = mpatches.Patch(color='none', label=f'NFE: {m_size*n_iters}')
    info2 = mpatches.Patch(color='none', label=f'N: {n_particles}')
    ax.legend(handles=handles + [info1, info2], fontsize=8, loc="upper right")

# NAIVE POINT PLOT
def naive_part_plot(ax, f=naive):
    _, _, Z = grid(res=300)
    # draw 2d array heatmap
    ax.imshow(Z, extent=[-32, 32, -32, 32], origin="lower", cmap="Blues", aspect="equal", norm=mcolors.PowerNorm(gamma=0.4,vmin=0,vmax=22))

    best_x, best_val, x_positions, f_vals, n_particles, variance = f(ackley, d=2)
    ax.scatter(x_positions[:,0], x_positions[:,1], color="orange", s=0.5) # plot individual points
    ax.scatter(best_x[0], best_x[1], color="red", s=12, label="min f(x)")

    ax.set_title("NAIVE ALGORITHM Particle Positions")
    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
    ax.set_xlim(-32, 32); ax.set_ylim(-32, 32)
    # add info patches to legend
    handles, labels = ax.get_legend_handles_labels()
    info1 = mpatches.Patch(color='none', label=f'NFE: {n_particles}')
    info2 = mpatches.Patch(color='none', label=f'N: {n_particles}')
    ax.legend(handles=handles + [info1, info2], fontsize=8, loc="upper right")


def main():
    # create blank canvas
    fig = plt.figure(figsize=(15,15))
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

    _, _, _, cbo_part_hist, _, _, _ = cbo(ackley, d=2, seed=0)
    _, _, _, var1_part_hist, _, _, _ = component_wise_diff(ackley, d=2, seed=0)
    _, _, _, var2_part_hist, _, _, _ = common_noise(ackley, d=2, seed=0)
    _, _, _, var3_part_hist, _, _, _ = random_mini_batch(ackley, d=2, seed=0)
    _, _, _, var13_part_hist, _, _, _ = comp_diff_random_batch(ackley, d=2, seed=0)

    particle_hist(ax1, cbo_part_hist)
    ax1.set_title("BC CBO Particle Trajectories")
    particle_hist(ax2, var1_part_hist)
    ax2.set_title("VARIANT 1: Particle Trajectories w/ Component-Wise Diffusion")
    particle_hist(ax3, var2_part_hist)
    ax3.set_title("VARIANT 2: Particle Trajectories w/ Component-Wise Diffusion & Common Noise")
    particle_hist(ax4, var3_part_hist, m_size=10)
    ax4.set_title("VARIANT 3: Particle Trajectories using Random Mini Batches")
    particle_hist(ax5, var13_part_hist, m_size=10)
    ax5.set_title("VARIANT 1+3: Particle Trajectories w/ Component-Wise Diffusion using Random Batching")

    naive_part_plot(ax6)
    naive_part_plot(ax7, f=naive_uniform)
    ax7.set_title("NAIVE ALGORITHM (UNIFORM GRID DISTR) Particle Positions")

    plt.tight_layout()
    plt.savefig("trajectories_plot.png", dpi=150)

if __name__ == "__main__":
    main()
