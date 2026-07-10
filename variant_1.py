import numpy as np
from bc_algorithms import ackley, consensus_point, cbo

# Variant 1 in Totzeck CBO: Component-wise diffusion
def component_wise_diff(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, seed=0):
    """ same drift as original cbo scheme, but now the diffusion is component-wise
        or adjusted for each particle individually"""

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
    best_val = np.inf
    # store variance/loss at each iteration
    variance_hist = []

    for _ in range(max_iters):
        # Consensus history
        f_vals = f(x) # re-run ackley every iteration
        c = consensus_point(x, f_vals, alpha)
        consensus_hist.append(c.copy())

        # Obj val history
        obj_val_hist.append(f_vals.min())

        # Best obj val history
        best_val = min(best_val, f_vals.min())
        best_val_hist.append(best_val)

        # Variance history
        mean = x.mean(axis=0) # expectation: average all part points
        v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance
        variance_hist.append(v)

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

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    return (x[best_idx], best_val_hist[-1], np.array(consensus_hist), np.array(particle_hist), np.array(obj_val_hist), np.array(best_val_hist), np.array(variance_hist))


if __name__ == "__main__": # runs when this is the main file of this execution
    print("CBO VARIANT 1")

    best_x, running_best, consensus_hist, _, obj_val_hist, _, _ = component_wise_diff(ackley, d=2, seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
    print(f"Iterations: {len(consensus_hist)}")
    print("True minimum: f(0,0) = 0")
