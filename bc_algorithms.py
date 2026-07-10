import numpy as np

# rng = np.random.default_rng(0) # set up seeded random num generator
# lo, hi = -32, 32
# x = rng.uniform(lo, hi, size=(50, 2)) # initialize uniform distr of all N particles at once

def ackley(x): # x = (x_1, x_2, ..., x_n)
    # d = dimension; N = number of particles in swarm
    # constants
    a = 20; b = 0.2; c = 2*np.pi
    d = x.shape[1]

    # results in one number for each particle
    sum_sq = np.sum(x**2, axis=1) / d # term Σxᵢ²/n (sum across row)
    sum_cos = np.sum(np.cos(c*x), axis = 1) / d # term Σcos(cxᵢ)/n

    # return result of Ackley function (np array of values)
    return -a*np.exp(-b*np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.exp(1)

# cbo on the first iteration
def naive(x, f): # x = (x_1, x_2, ..., x_n)
    f_vals = f(x)
    best_idx = np.argmin(f_vals) # index of lowest f val

    # return best part pos, best obj val, num of iterations
    return x[best_idx], f_vals[best_idx], len(f_vals)

def consensus_point(x, f_vals, alpha):
    weights = np.exp(-alpha*f_vals) # weight function
    weights = weights / weights.sum() # weigh each particle accordingly
    weighted_mean = np.sum(x*weights[:, None], axis=0) # sum down cols

    return weighted_mean # position of current global min

def cbo(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, seed=0):
    """ lambda = drift const; sigma = diffusion const; alpha = consensus sharpness
        dt = discretization step size for Euler-Maruyama
        tol = practical stand-in val (not acc calc val)
    """

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
        # Consensus history
        f_vals = f(x) # re-run ackley every iteration
        c = consensus_point(x, f_vals, alpha)
        consensus_hist.append(c.copy())

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

if __name__ == "__main__": # runs when this is the main file of this execution
    print("BASE CASE CBO")

    best_x, running_best, consensus_hist, _, obj_val_hist, _, _ = cbo(ackley, d=2, seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Ending/Best f(x): {obj_val_hist[-1]}")
    print(f"Iterations: {len(consensus_hist)}")
    print("True minimum: f(0,0) = 0")
