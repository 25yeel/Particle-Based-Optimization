import numpy as np
from bc_algorithms import ackley, consensus_point, cbo

# Variant 3 in Totzeck CBO: Random mini batches
def random_mini_batch(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, batch_size=10, seed=0):
    """ Same drift and diffusion as the base CBO scheme,
        BUT the consensus point is computed from random mini-batch of M particles
        rather than all N particles each iteration """

    # set up uniform distribution (Naive coverage)
    rng = np.random.default_rng(seed) # set up seeded random num generator
    lo, hi = domain
    x = rng.uniform(lo, hi, size=(n_particles, d))


    # HISTORY PLOTS
    # store all particle points at each iteration
    particle_hist = [] # [[[x1,x2],[x1,x2], ...]
                       #  [[x1,x2],[x1,x2], ...]
                       # ] trajectory shape(iterations, num parts, dim)
    # store consensus point at each iteration
    consensus_hist = [] # [[x1, x2], ...]
    # store best f(x) value at each iteration
    obj_val_hist = []
    # store RUNNING BEST f(x) value at each iteration
    best_val_hist = []
    running_best = np.inf
    # store variance/loss at each iteration
    variance_hist = []

    for _ in range(max_iters):
        # MINI-BATCH CONSENSUS POINT
        # draw M random particle indices without replacement
        batch_idx = rng.choice(n_particles, size=batch_size, replace=False)
        x_batch = x[batch_idx] # batch of points
        f_batch = f(x_batch) # obj values of batch

        # Consensus history
        c = consensus_point(x, f_batch, alpha) # consensus point evaluated from mini batch only
        consensus_hist.append(c.copy())

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

        # Particle history
        # EULER-MARUYAMA POS UPDATE
        diff = x - c # X̄ - consensus pt
        dist = np.linalg.norm(diff, axis=1, keepdims=True) # Euclidean dist |x-c|
        noise = rng.normal(size=(n_particles, d))
        dW = np.sqrt(dt)*noise # Brownian motion/random noise term

        # Pull each particle towards consensus
        x = x - lam*diff*dt + sigma*dist*dW # current pos - drift + diffusion

        particle_hist.append(x.copy())

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    return (x[best_idx], best_val_hist[-1], np.array(consensus_hist), np.array(particle_hist), np.array(obj_val_hist), np.array(best_val_hist), np.array(variance_hist))
