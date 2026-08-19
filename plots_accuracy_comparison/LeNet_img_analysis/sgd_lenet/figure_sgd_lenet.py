# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from plots_accuracy_comparison.sgd import sgd_simp

from LeNet.lenet_data import load_lenet_mnist
from LeNet.lenet_model import D_PARAMS, M_BATCH, make_batch_loss, lenet_forward


def main():
    X_train, y_train, X_test, y_test = load_lenet_mnist()

    # using make_batch_loss, instead of make_mnist_objective since is for grad-free cbo
    f_sgd = make_batch_loss(X_train, y_train)
    sgd_best_x, sgd_acc_hist = sgd_simp(f_sgd, d=D_PARAMS, num_epochs=25, tol=0.0, X_test=X_test, y_test=y_test, seed=0, n_data=len(X_train), m=M_BATCH, forward_fn=lenet_forward)

    fig, ax = plt.subplots(figsize=(8,6))
    for label, acc_hist in [("SGD", sgd_acc_hist)]:
        epochs = np.arange(1, len(acc_hist) + 1)
        ax.plot(epochs, acc_hist, label=label)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
    ax.set_title("Figure 6: SGD on MNIST (LeNet-5)")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure_sgd_lenet.png", dpi=150)


if __name__ == "__main__":
    main()
