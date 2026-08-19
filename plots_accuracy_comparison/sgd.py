# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from MNIST.mnist_data import N_TRAIN, load_mnist
from MNIST.mnist_model import mnist_forward, D_PARAMS, M_BATCH, make_mnist_objective

def sgd_simp(f, d, n_sims=100, lr=0.01, num_epochs=15, tol=0.0, X_test=None, y_test=None, seed=0, n_data=None, m=M_BATCH, forward_fn=mnist_forward):
    """ γ = learning rate/step size
        If n_data is given, f must accept (x, batch_idx) and return the mini-batch loss
        estimate for gradient
        If n_data is None, f(x) is called directly and the exact/full gradient is
        used every step """
    rng = np.random.default_rng(seed) # draw random starting pts (1 per simulation); also drives mini-batch draws
    # standard normal init, matching CBO's particle init (comp_diff_random_batch_simp) for a fair comparison
    x = torch.tensor(rng.normal(loc=0.0, scale=1.0, size=(n_sims, d)), dtype=torch.float32, requires_grad=True)

    # HISTORY PLOTS
    # store all sim positions at each iteration (mirrors cbo's particle_hist)
    # particle_hist = [x.detach().numpy().copy()]
    # store best f(x) value at each iteration
    # obj_val_hist = []
    # store RUNNING BEST f(x) value at each iteration
    # best_val_hist = []
    # running_best = np.inf

    active = np.ones(n_sims, dtype=bool) # boolean array to track which sims haven't converged yet

    epoch_step_size = N_TRAIN // m # epoch = n / m
    test_accuracy_hist = []
    def test_accuracy(x_star, X_test_data, y_test_data):
        """x_star is x_new[active] in SGD -> don't need to reshape anymore
            since forward_fn will accept the batch of n_particles directly"""
        """x_star = x_star.reshape(1,-1)"""
        probs = forward_fn(x_star, X_test_data) #(n_active, m, 10)
        """preds = probs.argmax(dim=-1).squeeze(0) # (4.5): g(x*, x̂)"""
        preds = probs.argmax(dim=-1) #(n_active, m)
        true_labels = y_test_data.argmax(dim=-1) #(m, )

        # indicator function
        return (preds == true_labels).float().mean().item() # (4.5): test accuracy function


    for i in tqdm(range(num_epochs*epoch_step_size)):
        # compute gradient via autograd; sims are independent so summing gives the correct per-row gradient
        if n_data is not None:
            batch_idx = rng.choice(n_data, size=m, replace=False) # random mini-batch of data indices
            f_vals = f(x, batch_idx) # mini-batch loss estimate L̂(x) -- SGD's noise
        else:
            f_vals = f(x)
        f_vals.sum().backward() # compute gradients
        grad = x.grad # ∂f_vals[i]/∂x[i] for each row i

        with torch.no_grad(): # update active rows only
            x_new = x.clone()
            x_new[active] = x[active] - lr*grad[active] # gradient descent step

        # Position history
        # particle_hist.append(x_new.detach().numpy().copy())

        # with torch.no_grad():
        #     # re-use the same mini-batch so the logged loss reflects the step just taken
        #     f_vals_new = (f(x_new, batch_idx) if n_data is not None else f(x_new)).detach().numpy()

        # Obj val history
        # obj_val_hist.append(f_vals_new.min())

        # Best obj val history
        # running_best = min(running_best, f_vals_new.min())
        # best_val_hist.append(running_best)

        # Freeze trajectories once they stop moving
        with torch.no_grad():
            step_size = torch.linalg.norm(x_new - x, dim=1).numpy()
        active = active & (step_size >= tol)


        if (i+1) % epoch_step_size == 0:
            # evaluate test accuracy
            x_star = x_new[active].detach().float()
            acc = test_accuracy(x_star, X_test, y_test)
            test_accuracy_hist.append(acc)

        x = x_new.detach().requires_grad_(True) # fresh leaf tensor for next iteration's autograd

        if not active.any(): # if all sims have converged
            break


    with torch.no_grad():
        if n_data is not None:
            batch_idx = rng.choice(n_data, size=m, replace=False)
            f_vals = f(x, batch_idx).detach().numpy()
        else:
            f_vals = f(x).detach().numpy()
    best_idx = np.argmin(f_vals)


    return (x[best_idx].detach().numpy(), np.array(test_accuracy_hist))
    # best x pos, running best obj val, particle hist, ending f vals, obj val hist, running best hist
    # return (particle_hist[-1][best_idx], best_val_hist[-1], np.array(particle_hist), np.array(f_vals), np.array(obj_val_hist), np.array(best_val_hist))
