import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from bc_algorithms import ackley, rastrigin

from plots_accuracy_comparison.var_1and3_MNISTuse import comp_diff_random_batch_simp
from plots_accuracy_comparison.var_argmin_MNISTuse import comp_diff_random_batch_argmin_simp
# variant without the last term in 2.5 -> set sigma=0

from MNIST.mnist_data import load_mnist
from MNIST.mnist_model import D_PARAMS, M_BATCH, make_mnist_objective

def main():
    X_train, y_train, X_test, y_test = load_mnist()

    # reference case (2.1, 4.6): full model, with noise, weighted avg consensus
    f_ref = make_mnist_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, ref_acc_hist = comp_diff_random_batch_simp(f_ref, d=D_PARAMS, num_epochs=200, tol=0.0, m=M_BATCH, X_test=X_test, y_test=y_test, seed=0)

    # # no noise variant: same model but with sigma=0
    f_nonoise = make_mnist_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, nonoise_acc_hist = comp_diff_random_batch_simp(f_nonoise, d=D_PARAMS, num_epochs=200, sigma=0.0, tol=0.0, m=M_BATCH, X_test=X_test, y_test=y_test, seed=0, kick_sigma=np.sqrt(0.1))

    # argmin variant: x̄* = argmin over the particle batch, instead of softmax average
    f_argmin = make_mnist_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, argmin_acc_hist = comp_diff_random_batch_argmin_simp(f_argmin, d=D_PARAMS, num_epochs=200, tol=0.0, m=M_BATCH, X_test=X_test, y_test=y_test, seed=0)

    fig, ax = plt.subplots(figsize=(8,6))
    for label, acc_hist in [("Reference (4.6)", ref_acc_hist),
                            # ("No noise", nonoise_acc_hist),
                            ("Argmin consensus", argmin_acc_hist)]:
        epochs = np.arange(1, len(acc_hist) + 1)
        ax.plot(epochs, acc_hist, label=label)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
    ax.set_title("Figure 6: CBO Variants on MNIST")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure_6_2.png", dpi=150)


if __name__ == "__main__":
    main()
