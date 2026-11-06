# LAST UPDATED 10/22/2026
import matplotlib.pyplot as plt
import numpy as np
import torch

from bc_algorithms import ackley, rastrigin, adamm, zo_grad_est

def single(f):
    return lambda x: float(f(x[None, :])[0])

def fo_grad(f):
    def g(x):
        xt = torch.tensor(x[None, :], dtype=torch.float64, requires_grad=True)
        f(xt).sum().backward()
        return xt.grad.numpy()[0]
    return g

f1 = single(ackley)
lo, hi, T, q = -32, 32, 1000, 10
starts = [np.array([20.0, 25.0]), np.array([-25.0, 10.0]), np.array([5.0, -28.0])]
zo_rng = np.random.default_rng(1)

# contour of Ackley
g = np.linspace(lo, hi, 400)
X, Y = np.meshgrid(g, g)
Z = ackley(np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(X.shape)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
colors = ["tab:red", "tab:blue", "tab:green"]

for ax, name in zip(axes[:2], ["First-order AdaMM", "ZO-AdaMM"]):
    ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax.plot(0, 0, "w*", ms=15, label="global min")
    ax.set_title(name); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

for x0, c in zip(starts, colors):
    _, h_fo = adamm(fo_grad(ackley), x0, T=T, alpha=1.0, beta1=0.9,
                    beta2=0.999, lo=lo, hi=hi, return_hist=True)
    _, h_zo = adamm(lambda x: zo_grad_est(f1, x, mu=1e-3, q=q, rng=zo_rng), x0,
                    T=T, alpha=1.0, beta1=0.9, beta2=0.3, lo=lo, hi=hi,
                    return_hist=True)

    for ax, h in zip(axes[:2], [h_fo, h_zo]):
        ax.plot(h[:, 0], h[:, 1], "-", color=c, lw=1, alpha=0.8)
        ax.plot(*h[0], "o", color=c, mec="w")      # start
        ax.plot(*h[-1], "s", color=c, mec="w")     # end

    axes[2].plot([f1(p) for p in h_fo], "-", color=c, label="FO")
    axes[2].plot([f1(p) for p in h_zo], "--", color=c, label="ZO")

axes[2].set_title("f(x) vs iteration (solid = FO, dashed = ZO)")
axes[2].set_xlabel("iteration"); axes[2].set_ylabel("Ackley f(x)")
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.show()
