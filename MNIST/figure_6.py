import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from bc_algorithms import ackley, rastrigin
from var_1and3_MNIST import comp_diff_random_batch # variant 2.1 with parameters 4.6
from var_argmin import comp_diff_random_batch_argmin # argmin variant 2.5
# variant without the last term in 2.5 -> set sigma=0

from mnist_data import load_mnist
from mnist_model import D_PARAMS, M_BATCH, make_mnist_objective

def test_accuracy_plot(test_acc_hist):
    pass

def main():
    # fig = plt.figure(figsize=(15,15))

    X_train, y_train, X_test, y_test = load_mnist()

    # reference case (2.1, 4.6): full model, with noise, weighted avg consensus
    f_ref = make_mnist_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, ref_acc_hist = comp_diff_random_batch(f_ref, d=D_PARAMS, tol=0.0, m=M_BATCH, X_test=X_test, y_test=y_test, seed=0)

    # no noise variant: samge model but with sigma=0
    f_nonoise = make_mnist_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, nonoise_acc_hist = comp_diff_random_batch(f_nonoise, d=D_PARAMS, sigma=0.0, tol=0.0, m=M_BATCH, X_test=X_test, y_test=y_test, seed=0)

    # argmin variant: x̄* = argmin over the particle batch, instead of softmax average
    f_argmin = make_mnist_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, argmin_acc_hist = comp_diff_random_batch_argmin(f_argmin, d=D_PARAMS, tol=0.0, m=M_BATCH, X_test=X_test, y_test=y_test, seed=0)

    fig, ax = plt.subplots(figsize=(8,6))
    for label, acc_hist in [("Reference (4.6)", ref_acc_hist),
                            ("No noise", nonoise_acc_hist),
                            ("Argmin consensus", argmin_acc_hist)]:
        epochs = np.arange(1, len(acc_hist) + 1)
        ax.plot(epochs, acc_hist, label=label)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1)
    ax.set_title("Figure 6: CBO variants on MNIST")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure_6.png", dpi=150)


if __name__ == "__main__":
    main()
