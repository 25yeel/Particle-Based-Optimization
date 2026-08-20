# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from bc_algorithms import ackley, consensus_point, cbo

# Variant 1 in Totzeck CBO: Component-wise diffusion + Random mini batches
def comp_diff_random_batch(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, batch_size=10, seed=0):
    """ Same drift and diffusion as the base CBO scheme,
        BUT the consensus point is computed from random mini-batch of M particles
        rather than all N particles each iteration """

    # set up uniform distribution (Naive coverage)
    rng = np.random.default_rng(seed) # set up seeded random num generator
    lo, hi = domain
    x = rng.uniform(lo, hi, size=(n_particles, d))

    f_vals = f(x)

    # HISTORY PLOTS
    # store all particle points at each iteration
    particle_hist = [x.copy()] # [[[x1,x2],[x1,x2], ...]
                       #  [[x1,x2],[x1,x2], ...]
                       # ] trajectory shape(iterations, num parts, dim)
    # store consensus point at each iteration
    batch_idx = rng.choice(n_particles, size=batch_size, replace=False)
    x_batch = x[batch_idx] # batch of points
    f_batch = f(x_batch) # obj values of batch
    c = consensus_point(x_batch, f_batch, alpha) # consensus point evaluated from mini batch only
    consensus_hist = [c.copy()] # [[x1, x2], ...]
    # store best f(x) value at each iteration
    obj_val_hist = [f_vals.min()]
    # store RUNNING BEST f(x) value at each iteration
    # sourced from the mini-batch (not the full swarm) so its NFE cost genuinely
    # matches batch_size*n_iters -- honest accounting for what's actually plotted
    best_val_hist = [f_batch.min()]
    running_best = np.inf
    # store variance/loss at each iteration
    mean = x.mean(axis=0) # expectation: average all part points
    v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance
    variance_hist = [v]

    for _ in range(max_iters):
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

        f_vals = f(x) # re-run ackley every iteration (full swarm -- diagnostic only, costs n_particles)

        # Obj val history
        obj_val_hist.append(f_vals.min())

        # Variance history
        mean = x.mean(axis=0) # expectation: average all part points
        v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance
        variance_hist.append(v)

        # MINI-BATCH CONSENSUS POINT
        # draw M random particle indices without replacement
        batch_idx = rng.choice(n_particles, size=batch_size, replace=False)
        x_batch = x[batch_idx] # batch of points
        f_batch = f(x_batch) # obj values of batch -- costs batch_size

        # Best obj val history -- sourced from the mini-batch (same evaluations used
        # for the consensus point below), so its NFE cost is honestly batch_size*n_iters
        running_best = min(running_best, f_batch.min())
        best_val_hist.append(running_best)

        # Consensus history
        c = consensus_point(x_batch, f_batch, alpha) # consensus point evaluated from mini batch only
        consensus_hist.append(c.copy())

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    return (x[best_idx], best_val_hist[-1], np.array(consensus_hist), np.array(particle_hist), np.array(obj_val_hist), np.array(best_val_hist), np.array(variance_hist))

if __name__ == "__main__": # runs when this is the main file of this execution
    print("CBO VARIANT 1+3")

    best_x, running_best, consensus_hist, _, obj_val_hist, _, _ = comp_diff_random_batch(ackley, d=2, seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
    print(f"Iterations: {len(consensus_hist)}")
    print("True minimum: f(0,0) = 0")
