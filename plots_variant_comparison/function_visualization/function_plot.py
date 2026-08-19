# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from bc_algorithms import ackley, rastrigin


# Ackley function in 1D
def ackley_1d(ax, res=1000, domain=(-32, 32)):
    lo, hi = domain
    lin = np.linspace(lo, hi, res)
    x = lin.reshape(-1, 1) # shape (res, 1) so ackley treats each point as d=1
    y = ackley(x)

    ax.plot(lin, y, color="#1D9E75", linewidth=1.8)
    ax.scatter(0, ackley(np.array([[0.0]]))[0], color="red", s=40, zorder=3, label="True min (0, 0)") # mark true min
    # local minima of ackley clearly separate from global min

    ax.set_title("Ackley Function in 1D")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Rastrigin function in 1D
def rastrigin_1d(ax, res=1000, domain=(-5.12, 5.12)):
    lo, hi = domain
    lin = np.linspace(lo, hi, res)
    x = lin.reshape(-1, 1) # shape (res, 1) so rastrigin treats each point as d=1
    y = rastrigin(x)

    ax.plot(lin, y, color="#5A6DD8", linewidth=1.8)
    ax.scatter(0, rastrigin(np.array([[0.0]]))[0], color="red", s=40, zorder=3, label="True min (0, 0)") # mark true min
    # local minima of rastrigin are very close to the global min

    ax.set_title("Rastrigin Function in 1D")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    # create blank canvas
    fig = plt.figure(figsize=(15,15))
    fig.suptitle("Functions illustrated in 1D", fontsize=14, fontweight="bold")

    # 2x2 figure panels
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    # plot graphs
    ackley_1d(ax1)
    rastrigin_1d(ax2)

    plt.tight_layout()
    plt.savefig("function_plot.png", dpi=150) # save png to folder

if __name__ == "__main__":
    main()
