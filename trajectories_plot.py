import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bc_algorithms import ackley, cbo
from variant_1 import component_wise_diff
from variant_2 import common_noise
from variant_3 import random_mini_batch


def grid(res):
    # create grid
    lin = np.linspace(-32, 32, res)
    X, Y = np.meshgrid(lin, lin)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

    # evaluate function at all grid points, correspond to grid coord
    Z = ackley(grid).reshape(res, res)

    return X, Y, Z

# TRAJECTORY PLOT
def particle_hist(ax, particle_hist):
    _, _, Z = grid(res=300)

    # draw 2d array heatmap
    ax.imshow(Z, extent=[-32, 32, -32, 32], origin="lower", cmap="Blues", aspect="equal", norm=mcolors.PowerNorm(gamma=0.4,vmin=0,vmax=22))

    n_particles = particle_hist.shape[1]

    for i in range(n_particles):
        traj = particle_hist[:,i,:] # draw one trajectory line per particle
        x_1 = traj[:,0]
        x_2 = traj[:,1]
        ax.plot(x_1, x_2, color="white", alpha=0.15, linewidth=1.0)
    # plot individual points at iteration 0
    ax.scatter(particle_hist[0,:,0], particle_hist[0,:,1], color="orange", s=12, zorder=3, label="Start")
    # plot individual points at iteration 1
    ax.scatter(particle_hist[-1,:,0], particle_hist[-1,:,1], color="red", s=12, zorder=3, label="End")

    # styling
    # ax.set_title("Particle Trajectories")
    ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
    ax.set_xlim(-32, 32); ax.set_ylim(-32, 32)
    ax.legend(fontsize=7, loc="upper right")

def main():
    # create blank canvas
    fig = plt.figure(figsize=(13,10))
    fig.suptitle("CBO on the 2D Ackley Function", fontsize=14, fontweight="bold")

    # 2x2 figure panels
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    # plot graphs
    _, _, _, cbo_part_hist, _, _, _ = cbo(ackley, d=2, seed=0)
    _, _, _, var1_part_hist, _, _, _ = component_wise_diff(ackley, d=2, seed=0)
    _, _, _, var2_part_hist, _, _, _ = common_noise(ackley, d=2, seed=0)
    _, _, _, var3_part_hist, _, _, _ = random_mini_batch(ackley, d=2, seed=0)
    particle_hist(ax1, cbo_part_hist)
    ax1.set_title("BC CBO Particle Trajectories")
    particle_hist(ax2, var1_part_hist)
    ax2.set_title("VARIANT 1 Particle Trajectories w/ Component-Wise Diffusion")
    particle_hist(ax3, var2_part_hist)
    ax3.set_title("VARIANT 2 Particle Trajectories w/ Component-Wise Diffusion & Common Noise")
    particle_hist(ax4, var3_part_hist)
    ax4.set_title("VARIANT 3 Particle Trajectories using Random Mini Batches")


    plt.tight_layout()
    plt.savefig("trajectories_plot.png", dpi=150)

if __name__ == "__main__":
    main()
