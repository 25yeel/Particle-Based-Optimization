# LAST UPDATED 11/6/2026
# ZO-AdaMM (Chen et al., Algorithm 1) on MNIST, run on a batch of particles.
# Drop-in counterpart to var_1and3_MNISTuse.py / var_argmin_MNISTuse.py / sgd.py:
# same init, same epoch definition, same (best_x, test_accuracy_hist) return.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from bc_algorithms import zo_grad_est_batch
from MNIST.mnist_data import N_TRAIN
from MNIST.mnist_model import mnist_forward, M_BATCH


def zo_adamm_simp(f, d, n_particles=10, q=10, mu=1e-2, alpha=0.2, beta1=0.9, beta2=0.3, v0=1e-5,
                  num_epochs=15, m=M_BATCH, n_data=N_TRAIN, X_test=None, y_test=None, seed=0,
                  forward_fn=mnist_forward):
    """ ZO-AdaMM on a batch of n_particles independent points.

        f : torch loss f(x_t, batch_idx) -> (N,), e.g. make_batch_loss(X_train, y_train)
            (NOT make_*_objective: that one redraws a mini-batch on every call, and the ZO
            estimate needs f(x + mu*u) and f(x) on the SAME batch, otherwise batch noise
            gets multiplied by d/mu and swamps the gradient)
        Per iteration: draw one mini-batch, estimate the gradient with q random directions
        (zo_grad_est_batch), then do the AMSGrad-style AdaMM update with alpha_t = alpha/sqrt(t).
        NFE per iteration = n_particles * (q + 1) mini-batch loss evaluations. """

    rng = np.random.default_rng(seed)
    # standard normal init, matching CBO / SGD for a fair comparison
    x = rng.normal(loc=0.0, scale=1.0, size=(n_particles, d))

    m_t = np.zeros_like(x)          # first moment (momentum)
    v_t = np.zeros_like(x)          # second moment
    vhat = np.full_like(x, v0)      # max of second moments (never decreases)

    epoch_step_size = n_data // m   # epoch = n / m
    test_accuracy_hist = []
    loss_hist = []                  # mean mini-batch loss of the best particle, per epoch

    def f_np(z, batch_idx):         # numpy (N, d) -> numpy (N,) on a FIXED mini-batch
        with torch.no_grad():
            return f(torch.from_numpy(z).float(), batch_idx).numpy().astype(np.float64)

    def test_accuracy(x_star, X_test_data, y_test_data):
        x_star = torch.from_numpy(x_star).float().reshape(1, -1)
        with torch.no_grad():
            probs = forward_fn(x_star, X_test_data)
        preds = probs.argmax(dim=-1).squeeze(0)
        true_labels = y_test_data.argmax(dim=-1)
        return (preds == true_labels).float().mean().item()

    epoch_losses = []
    f_vals = None
    for t in tqdm(range(1, num_epochs * epoch_step_size + 1)):
        batch_idx = rng.choice(n_data, size=m, replace=False)
        f_batch = lambda z: f_np(z, batch_idx)  # same batch for f(x) and every f(x + mu*u)

        f_vals = f_batch(x)
        epoch_losses.append(f_vals.min())

        g = zo_grad_est_batch(f_batch, x, f_vals, mu=mu, q=q, rng=rng)
        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g**2
        vhat = np.maximum(vhat, v_t)
        x = x - (alpha / np.sqrt(t)) * m_t / np.sqrt(vhat)

        if t % epoch_step_size == 0:
            # evaluate the particle with the lowest loss on this iteration's mini-batch
            best_idx = np.argmin(f_vals)
            acc = test_accuracy(x[best_idx], X_test, y_test)
            test_accuracy_hist.append(acc)
            loss_hist.append(np.mean(epoch_losses))
            epoch_losses = []
            tqdm.write(f"epoch {t // epoch_step_size}: best-particle train loss {loss_hist[-1]:.4f}, test acc {acc:.4f}")

    best_idx = np.argmin(f_np(x, rng.choice(n_data, size=m, replace=False)))
    return (x[best_idx], np.array(test_accuracy_hist), np.array(loss_hist))


def main():
    import argparse
    import matplotlib.pyplot as plt
    from LeNet.lenet_data import load_lenet_mnist
    from LeNet.lenet_model import D_PARAMS, make_batch_loss, lenet_forward

    p = argparse.ArgumentParser(description="ZO-AdaMM on LeNet-5 / MNIST")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--particles", type=int, default=10)
    p.add_argument("--q", type=int, default=10)
    p.add_argument("--mu", type=float, default=1e-2)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output png (default: LeNet_img_analysis/zo_adamm_lenet/...)")
    args = p.parse_args()

    # reuse the already-downloaded copy instead of re-downloading into the cwd
    X_train, y_train, X_test, y_test = load_lenet_mnist(root=str(ROOT / "LeNet" / "mnist_data"))
    f = make_batch_loss(X_train, y_train)

    best_x, acc_hist, loss_hist = zo_adamm_simp(
        f, d=D_PARAMS, n_particles=args.particles, q=args.q, mu=args.mu, alpha=args.alpha,
        beta1=args.beta1, beta2=args.beta2, num_epochs=args.epochs, m=M_BATCH, n_data=len(X_train),
        X_test=X_test, y_test=y_test, seed=args.seed, forward_fn=lenet_forward)

    out = Path(args.out) if args.out else (
        ROOT / "plots_accuracy_comparison" / "LeNet_img_analysis" / "zo_adamm_lenet" /
        f"figure_zo_adamm_lenet_N{args.particles}_q{args.q}_a{args.alpha}_{args.epochs}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), acc_hist)  # keep raw accuracies for later CBO/SGD overlay

    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = np.arange(1, len(acc_hist) + 1)
    ax.plot(epochs, acc_hist, label=f"ZO-AdaMM (N={args.particles}, q={args.q})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
    ax.set_title("ZO-AdaMM on MNIST (LeNet-5)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
