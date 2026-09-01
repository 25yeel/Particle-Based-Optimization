# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from plots_accuracy_comparison.var_1and3_MNISTuse import comp_diff_random_batch_simp
from plots_accuracy_comparison.var_argmin_MNISTuse import comp_diff_random_batch_argmin_simp

from LeNet.lenet_data import load_lenet_mnist
from LeNet.lenet_model import D_PARAMS, M_BATCH, make_lenet_objective, lenet_forward

# SAME AS figure_6.py, EXCEPT forward_fn=lenet_forward instead of mnist_forward
def main():
    X_train, y_train, X_test, y_test = load_lenet_mnist()

    # reference case (2.1, 4.6): full model, with noise, weighted avg consensus
    f_ref = make_lenet_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, ref_acc_hist = comp_diff_random_batch_simp(f_ref, d=D_PARAMS, lam=0.8, sigma=0.2, alpha=30.0, num_epochs=500, tol=0.0, X_test=X_test, y_test=y_test, seed=0, forward_fn=lenet_forward)

    # no noise variant: same model but with sigma=0
    f_nonoise = make_lenet_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, nonoise_acc_hist = comp_diff_random_batch_simp(f_nonoise, d=D_PARAMS, lam=0.8, sigma=0.0, num_epochs=20, tol=0.0, X_test=X_test, y_test=y_test, seed=0, kick_sigma=0.2, forward_fn=lenet_forward)

    # argmin variant: x̄* = argmin over the particle batch, instead of softmax average
    f_argmin = make_lenet_objective(X_train, y_train, m=M_BATCH, seed=0)
    *_, argmin_acc_hist = comp_diff_random_batch_argmin_simp(f_argmin, d=D_PARAMS, lam=0.8, sigma=0.2, num_epochs=20, tol=0.0, X_test=X_test, y_test=y_test, seed=0, forward_fn=lenet_forward)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, acc_hist in [("Reference (4.6)", ref_acc_hist)]:
                            # ("No noise", nonoise_acc_hist),
                            # ("Argmin consensus", argmin_acc_hist)]:
        epochs = np.arange(1, len(acc_hist) + 1)
        ax.plot(epochs, acc_hist, label=label)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
    ax.set_title("CBO Variants on MNIST (LeNet-5)")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure_lenet_sig02_lam08_500.png", dpi=150)


if __name__ == "__main__":
    main()
