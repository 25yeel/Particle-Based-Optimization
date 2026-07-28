import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from bc_algorithms import ackley, consensus_point, cbo
from mnist_data import N_TRAIN, load_mnist
from mnist_model import mnist_forward, D_PARAMS, M_BATCH, make_mnist_objective


def comp_diff_random_batch_argmin(f, d, domain=(-32,32), n_particles=100, lam=1.0, sigma=np.sqrt(0.1), dt=0.1, max_iters=500, tol=0.0, M=10, m=50, X_test=None, y_test=None, seed=0):
    """ Same as comp_diff_random_batch, but x̄* is the argmin of the batch (2.5)
        instead of the softmax-weighted average (2.3) -> no 'alpha' needed here """

    # set up standard normal distribution
    rng = np.random.default_rng(seed) # set up seeded random num generator
    x = rng.normal(loc=0.0, scale=1.0, size=(n_particles, d))  # each particle ~ N(0, I_d)

    f_vals = f(x)

    # HISTORY PLOTS
    # store all particle points at each iteration
    particle_hist = [x.copy()] # [[[x1,x2],[x1,x2], ...]
                       #  [[x1,x2],[x1,x2], ...]
                       # ] trajectory shape(iterations, num parts, dim)
    # store consensus point at each iteration
    batch_idx = rng.choice(n_particles, size=M, replace=False)
    x_batch = x[batch_idx] # batch of points
    f_batch = f(x_batch) # obj values of batch
    # CHANGE IN CALCULATION OF CONSENSUS POINTS -> ALPHA IS DROPPED
    # c = consensus_point(x_batch, f_batch, alpha) # consensus point evaluated from mini batch only
    c = x_batch[np.argmin(f_batch)]
    consensus_hist = [c.copy()] # [[x1, x2], ...]
    # store best f(x) value at each iteration
    obj_val_hist = [f_vals.min()]
    # store RUNNING BEST f(x) value at each iteration
    best_val_hist = [f_vals.min()]
    running_best = np.inf
    # store variance/loss at each iteration
    mean = x.mean(axis=0) # expectation: average all part points
    v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance
    variance_hist = [v]
    # store test accuracy at each epoch
    epoch_step_size = N_TRAIN // m # epoch = n / m
    test_accuracy_hist = []
    def test_accuracy(x_star, X_test_data, y_test_data):
        x_star = x_star.reshape(1,-1)
        probs = mnist_forward(x_star, X_test_data)
        preds = probs.argmax(dim=-1).squeeze(0) # (4.5): g(x*, x̂)
        true_labels = y_test_data.argmax(dim=-1)

        # indicator function
        return (preds == true_labels).float().mean().item() # (4.5): test accuracy function

    for i in range(max_iters):
        # Particle history
        # EULER-MARUYAMA POS UPDATE
        diff = x - c # X̄ - consensus pt
        """ WE NO LONGER NEED TO COMPUTE EUCLIDEAN DIST
            with dist we mult every dW by the same scalar (all d directions scaled identically)
            BUT NOW each col k col k of hte update is diff[:,k] * dW[:,k]
            scaled by that dim's own dist to consensus """
        # dist = np.linalg.norm(diff, axis=1, keepdims=True) # Euclidean dist |x-c|
        noise = rng.normal(size=(n_particles, d))
        dW = np.sqrt(dt)*noise # Brownian motion/random noise term

        # Pull each particle towards consensus
        x = x - lam*diff*dt + sigma*diff*dW # current pos - drift + diffusion

        particle_hist.append(x.copy())

        f_vals = f(x) # re-run ackley every iteration

        # Obj val history
        obj_val_hist.append(f_vals.min())

        # Best obj val history
        running_best = min(running_best, f_vals.min())
        best_val_hist.append(running_best)

        # Variance history
        mean = x.mean(axis=0) # expectation: average all part points
        v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance
        variance_hist.append(v)

        # MINI-BATCH CONSENSUS POINT
        # draw M random particle indices without replacement
        batch_idx = rng.choice(n_particles, size=M, replace=False)
        x_batch = x[batch_idx] # batch of points
        f_batch = f(x_batch) # obj values of batch

        # Consensus history
        # CHANGE IN CALCULATION OF CONSENSUS POINTS
        c = x_batch[np.argmin(f_batch)] # consensus point evaluated from mini batch only
        consensus_hist.append(c.copy())

        if (i+1) % epoch_step_size == 0:
            # evaluate test accuracy
            x_star = torch.from_numpy(c).float()
            acc = test_accuracy(x_star, X_test, y_test)
            test_accuracy_hist.append(acc)

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    return (x[best_idx], best_val_hist[-1], np.array(consensus_hist), np.array(particle_hist), np.array(obj_val_hist), np.array(best_val_hist), np.array(variance_hist), np.array(test_accuracy_hist))

if __name__ == "__main__": # runs when this is the main file of this execution
    print("CBO VARIANT 1+3 (ARGMIN CONSENSUS PT) APPLIED TO MNIST")

    X_train, y_train, X_test, y_test = load_mnist()
    f = make_mnist_objective(X_train, y_train, m=M_BATCH, seed=0)
    best_x, running_best, consensus_hist, part_hist, obj_val_hist, best_val_hist, var_hist, test_acc_hist = comp_diff_random_batch_argmin(f, d=D_PARAMS, m=M_BATCH, X_test=X_test, y_test=y_test, seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
    print(f"Iterations: {len(consensus_hist)}")
    print("True minimum: f(0,0) = 0")
