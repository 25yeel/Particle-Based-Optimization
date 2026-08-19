# LAST UPDATED 8/19/2026
import numpy as np
from bc_algorithms import ackley, consensus_point

NFE_comp_benchmark = 1000

def naive_fixedNFE(f, d, domain=(-32,32), seed=0):
    rng = np.random.default_rng(seed) # set up seeded random num generator
    lo, hi = domain
    x = rng.uniform(lo, hi, size=(NFE_comp_benchmark, d)) # x = (x_1, x_2, ..., x_n)

    f_vals = f(x)
    best_idx = np.argmin(f_vals) # index of lowest f val

    # Variance history
    mean = x.mean(axis=0) # expectation: average all part points
    v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance

    best_x = x[best_idx]; best_obj_val = f_vals[best_idx]; x_positions = np.array(x); f_vals = np.array(f_vals); n_particles = len(f_vals)

    if __name__ == "__main__":
        print("---------------------------------------")
        print("NAIVE ALGORITHM RESULTS")
        print(f"Best x position found: {best_x}")
        print(f"Best f(x): {best_obj_val:.6f}")
        print(f"num particles: {n_particles}")
        print(f"Variance: {v}")
        print(f"NFE: {NFE_comp_benchmark}")

    return (best_x, best_obj_val, x_positions, f_vals, n_particles, v)

def naive_uniform_fixedNFE(f, d, domain=(-32,32)):
    # generate uniform grid of points
    lo, hi = domain
    pts_per_dim = round(NFE_comp_benchmark ** (1/d)) # same num of pts on each axis
    lin = np.linspace(lo, hi, pts_per_dim)
    mesh = np.meshgrid(*([lin] * d))
    x = np.stack([m.ravel() for m in mesh], axis=1) # shape: (pts_per_dim**d, d)

    f_vals = f(x)
    best_idx = np.argmin(f_vals) # index of lowest f val

    # Variance history
    mean = x.mean(axis=0) # expectation: average all part points
    v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance

    best_x = x[best_idx]; best_obj_val = f_vals[best_idx]; x_positions = np.array(x); f_vals = np.array(f_vals); n_particles = len(f_vals)

    if __name__ == "__main__":
        print("---------------------------------------")
        print("NAIVE ALGORITHM (UNIFORM GRID DISTR) RESULTS")
        print(f"Best x position found: {best_x}")
        print(f"Best f(x): {best_obj_val:.6f}")
        print(f"num particles: {n_particles}")
        print(f"Variance: {v}")
        print(f"NFE: {len(f_vals)}")

    return (best_x, best_obj_val, x_positions, f_vals, n_particles, v)

def cbo_fixedNFE(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, seed=0):
    """ lambda = drift const; sigma = diffusion const; alpha = consensus sharpness
        dt = discretization step size for Euler-Maruyama
        tol = practical stand-in val (not acc calc val)
    """

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
    c = consensus_point(x, f_vals, alpha)
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

    for _ in range(max_iters):
        # EULER-MARUYAMA POS UPDATE
        diff = x - c # X̄ - consensus pt
        dist = np.linalg.norm(diff, axis=1, keepdims=True) # Euclidean dist |x-c|
        noise = rng.normal(size=(n_particles, d))
        dW = np.sqrt(dt)*noise # Brownian motion/random noise term

        # Pull each particle towards consensus
        x = x - lam*diff*dt + sigma*dist*dW # current pos - drift + diffusion

        # Particle history
        particle_hist.append(x.copy())

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

        n_iters = len(consensus_hist)
        NFE = n_particles*n_iters
        if NFE >= NFE_comp_benchmark: # benchmark NFE
            break

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)


    best_x = x[best_idx]; running_best = best_val_hist[-1]; consensus_hist = np.array(consensus_hist); particle_hist = np.array(particle_hist); obj_val_hist = np.array(obj_val_hist); best_val_hist = np.array(best_val_hist); var_hist = np.array(variance_hist)

    if __name__ == "__main__":
        print("---------------------------------------")
        print("CBO")
        print(f"Best x position found: {best_x}")
        print(f"Running Best f(x): {running_best:.6f}")
        print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
        print(f"Iterations: {len(consensus_hist)}")
        print(f"NFE: {NFE}")

    return (best_x, running_best, consensus_hist, particle_hist, obj_val_hist, best_val_hist, var_hist)

# Variant 1 in Totzeck CBO: Component-wise diffusion
def component_wise_diff_fixedNFE(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, seed=0):
    """ same drift as original cbo scheme, but now the diffusion is component-wise
        or adjusted for each particle individually"""

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
    c = consensus_point(x, f_vals, alpha)
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

        n_iters = len(consensus_hist)
        NFE = n_particles*n_iters
        if NFE >= NFE_comp_benchmark: # benchmark NFE
            break

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    best_x = x[best_idx]; running_best = best_val_hist[-1]; consensus_hist = np.array(consensus_hist); particle_hist = np.array(particle_hist); obj_val_hist = np.array(obj_val_hist); best_val_hist = np.array(best_val_hist); var_hist = np.array(variance_hist)

    if __name__ == "__main__":
        print("---------------------------------------")
        print("CBO VARIANT 1")
        print(f"Best x position found: {best_x}")
        print(f"Running Best f(x): {running_best:.6f}")
        print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
        print(f"Iterations: {len(consensus_hist)}")
        print(f"NFE: {NFE}")

    return (best_x, running_best, consensus_hist, particle_hist, obj_val_hist, best_val_hist, var_hist)


# Variant 2 in Totzeck CBO: Component-wise COMMON diffusion
def common_noise_fixedNFE(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, seed=0):
    """ same drift as original cbo scheme, but now the diffusion is component-wise
        or adjusted for each particle individually"""

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
    c = consensus_point(x, f_vals, alpha)
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

    for _ in range(max_iters):
        # Particle history
        # EULER-MARUYAMA POS UPDATE
        diff = x - c # X̄ - consensus pt
        """ VARIANT 1: WE NO LONGER NEED TO COMPUTE EUCLIDEAN DIST
            with dist we mult every dW by the same scalar (all d directions scaled identically)
            BUT NOW each col k col k of hte update is diff[:,k] * dW[:,k]
            scaled by that dim's own dist to consensus """
        # dist = np.linalg.norm(diff, axis=1, keepdims=True) # Euclidean dist |x-c|
        """ VARIANT 2: COMMON NOISE
            ONE DRAW FOR NOISE, shared by all particles instead of random for each
        """
        noise = rng.normal(size=(1, d))
        dW = np.sqrt(dt)*noise # Brownian motion/random noise term

        # Pull each particle towards consensus
        x = x - lam*diff*dt + sigma*diff*dW # current pos - drift + diffusion

        particle_hist.append(x.copy())

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

        n_iters = len(consensus_hist)
        NFE = n_particles*n_iters
        if NFE >= NFE_comp_benchmark: # benchmark NFE
            break

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    best_x = x[best_idx]; running_best = best_val_hist[-1]; consensus_hist = np.array(consensus_hist); particle_hist = np.array(particle_hist); obj_val_hist = np.array(obj_val_hist); best_val_hist = np.array(best_val_hist); var_hist = np.array(variance_hist)

    if __name__ == "__main__":
        print("---------------------------------------")
        print("CBO VARIANT 2")
        print(f"Best x position found: {best_x}")
        print(f"Running Best f(x): {running_best:.6f}")
        print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
        print(f"Iterations: {len(consensus_hist)}")
        print(f"NFE: {NFE}")

    return (best_x, running_best, consensus_hist, particle_hist, obj_val_hist, best_val_hist, var_hist)

# Variant 3 in Totzeck CBO: Random mini batches
def random_mini_batch_fixedNFE(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, batch_size=10, seed=0):
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
    best_val_hist = [f_vals.min()]
    running_best = np.inf
    # store variance/loss at each iteration
    mean = x.mean(axis=0) # expectation: average all part points
    v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance
    variance_hist = [v]

    for _ in range(max_iters):
        # Particle history
        # EULER-MARUYAMA POS UPDATE
        diff = x - c # X̄ - consensus pt
        dist = np.linalg.norm(diff, axis=1, keepdims=True) # Euclidean dist |x-c|
        noise = rng.normal(size=(n_particles, d))
        dW = np.sqrt(dt)*noise # Brownian motion/random noise term

        # Pull each particle towards consensus
        x = x - lam*diff*dt + sigma*dist*dW # current pos - drift + diffusion

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
        batch_idx = rng.choice(n_particles, size=batch_size, replace=False)
        x_batch = x[batch_idx] # batch of points
        f_batch = f(x_batch) # obj values of batch

        # Consensus history
        c = consensus_point(x_batch, f_batch, alpha) # consensus point evaluated from mini batch only
        consensus_hist.append(c.copy())

        n_iters = len(consensus_hist)
        NFE = batch_size*n_iters
        if NFE >= NFE_comp_benchmark: # benchmark NFE
            break

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    best_x = x[best_idx]; running_best = best_val_hist[-1]; consensus_hist = np.array(consensus_hist); particle_hist = np.array(particle_hist); obj_val_hist = np.array(obj_val_hist); best_val_hist = np.array(best_val_hist); var_hist = np.array(variance_hist)

    if __name__ == "__main__":
        print("---------------------------------------")
        print("CBO VARIANT 3")
        print(f"Best x position found: {best_x}")
        print(f"Running Best f(x): {running_best:.6f}")
        print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
        print(f"Iterations: {len(consensus_hist)}")
        print(f"NFE: {NFE}")

    return (best_x, running_best, consensus_hist, particle_hist, obj_val_hist, best_val_hist, var_hist)

# Variant 1 in Totzeck CBO: Component-wise diffusion + Random mini batches
def comp_diff_random_batch_fixedNFE(f, d, domain=(-32,32), n_particles=50, lam=1.0, sigma=1.0, alpha=30.0, dt=0.05, max_iters=500, tol=1e-5, batch_size=10, seed=0):
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
    best_val_hist = [f_vals.min()]
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
        batch_idx = rng.choice(n_particles, size=batch_size, replace=False)
        x_batch = x[batch_idx] # batch of points
        f_batch = f(x_batch) # obj values of batch

        # Consensus history
        c = consensus_point(x_batch, f_batch, alpha) # consensus point evaluated from mini batch only
        consensus_hist.append(c.copy())

        n_iters = len(consensus_hist)
        NFE = batch_size*n_iters
        if NFE >= NFE_comp_benchmark: # benchmark NFE
            break

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    best_x = x[best_idx]; running_best = best_val_hist[-1]; consensus_hist = np.array(consensus_hist); particle_hist = np.array(particle_hist); obj_val_hist = np.array(obj_val_hist); best_val_hist = np.array(best_val_hist); var_hist = np.array(variance_hist)

    if __name__ == "__main__":
        print("---------------------------------------")
        print("CBO VARIANT 1&3")
        print(f"Best x position found: {best_x}")
        print(f"Running Best f(x): {running_best:.6f}")
        print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
        print(f"Iterations: {len(consensus_hist)}")
        print(f"NFE: {NFE}")

    return (best_x, running_best, consensus_hist, particle_hist, obj_val_hist, best_val_hist, var_hist)

if __name__ == "__main__": # runs when this is the main file of this execution
    naive_fixedNFE(ackley, d=2)
    naive_uniform_fixedNFE(ackley, d=2)
    cbo_fixedNFE(ackley, d=2)
    component_wise_diff_fixedNFE(ackley, d=2)
    common_noise_fixedNFE(ackley, d=2)
    random_mini_batch_fixedNFE(ackley, d=2)
    comp_diff_random_batch_fixedNFE(ackley, d=2)
    print("---------------------------------------")
    print("True minimum: f(0,0) = 0")
