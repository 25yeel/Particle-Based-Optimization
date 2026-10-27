# LAST UPDATED 10/22/2026
import numpy as np
import torch

# rng = np.random.default_rng(0) # set up seeded random num generator
# lo, hi = -32, 32
# x = rng.uniform(lo, hi, size=(50, 2)) # initialize uniform distr of all N particles at once

def ackley(x): # x = (x_1, x_2, ..., x_n)
    # d = dimension; N = number of particles in swarm
    # constants
    a = 20; b = 0.2; c = 2*np.pi
    d = x.shape[1]
    lib = torch if isinstance(x, torch.Tensor) else np # dispatch so autograd works when x is a tensor (see sgd)

    # results in one number for each particle
    sum_sq = (x**2).sum(axis=1) / d # term Σxᵢ²/n (sum across row)
    sum_cos = lib.cos(c*x).sum(axis=1) / d # term Σcos(cxᵢ)/n

    # return result of Ackley function (array of values)
    return -a*lib.exp(-b*lib.sqrt(sum_sq)) - lib.exp(sum_cos) + a + np.exp(1)

def rastrigin(x, B=0.0, C=0.0): # x = (x_1, x_2, ..., x_n)
    # B = argmin L(x); C = min L(x)

    shifted = x - B
    d = shifted.shape[-1] # d = dimension
    lib = torch if isinstance(x, torch.Tensor) else np # account for cbo (np) and sgd (torch)
    terms = shifted**2 - 10*lib.cos(2*np.pi*shifted) + 10

    # return result of Rastrigin function (array of values)
    return terms.sum(axis=-1) / d + C

def sgd(f, d, domain=(-32,32), n_sims=100, lr=0.01, max_iters=10_000, tol=1e-5, seed=0, n_data=None, batch_size=None):
    """ γ = learning rate/step size
        If n_data is given, f must accept (x, batch_idx) and return the mini-batch loss
        estimate for gradient
        If n_data is None, f(x) is called directly and the exact/full gradient is
        used every step """
    rng = np.random.default_rng(seed) # draw random starting pts (1 per simulation); also drives mini-batch draws
    lo, hi = domain
    x = torch.tensor(rng.uniform(lo, hi, size=(n_sims, d)), requires_grad=True)

    # HISTORY PLOTS
    # store all sim positions at each iteration (mirrors cbo's particle_hist)
    particle_hist = [x.detach().numpy().copy()]
    # store best f(x) value at each iteration
    obj_val_hist = []
    # store RUNNING BEST f(x) value at each iteration
    best_val_hist = []
    running_best = np.inf

    active = np.ones(n_sims, dtype=bool) # boolean array to track which sims haven't converged yet

    for _ in range(max_iters):
        # compute gradient via autograd; sims are independent so summing gives the correct per-row gradient
        if n_data is not None:
            batch_idx = rng.choice(n_data, size=batch_size, replace=False) # random mini-batch of data indices
            f_vals = f(x, batch_idx) # mini-batch loss estimate L̂(x) -- SGD's noise
        else:
            f_vals = f(x)
        f_vals.sum().backward() # compute gradients
        grad = x.grad # ∂f_vals[i]/∂x[i] for each row i

        with torch.no_grad(): # update active rows only
            x_new = x.clone()
            x_new[active] = x[active] - lr*grad[active] # gradient descent step

        # Position history
        particle_hist.append(x_new.detach().numpy().copy())

        with torch.no_grad():
            # re-use the same mini-batch so the logged loss reflects the step just taken
            f_vals_new = (f(x_new, batch_idx) if n_data is not None else f(x_new)).detach().numpy()

        # Obj val history
        obj_val_hist.append(f_vals_new.min())

        # Best obj val history
        running_best = min(running_best, f_vals_new.min())
        best_val_hist.append(running_best)

        # Freeze trajectories once they stop moving
        with torch.no_grad():
            step_size = torch.linalg.norm(x_new - x, dim=1).numpy()
        active = active & (step_size >= tol)

        x = x_new.detach().requires_grad_(True) # fresh leaf tensor for next iteration's autograd

        if not active.any(): # if all sims have converged
            break


    with torch.no_grad():
        if n_data is not None:
            batch_idx = rng.choice(n_data, size=batch_size, replace=False)
            f_vals = f(x, batch_idx).detach().numpy()
        else:
            f_vals = f(x).detach().numpy()
    best_idx = np.argmin(f_vals)

    # best x pos, running best obj val, particle hist, ending f vals, obj val hist, running best hist
    return (particle_hist[-1][best_idx], best_val_hist[-1], np.array(particle_hist), np.array(f_vals), np.array(obj_val_hist), np.array(best_val_hist))


def consensus_point(x, f_vals, alpha):
    weights = np.exp(-alpha*(f_vals - f_vals.min())) # weight function
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

        # Check how spread out the particles are (relative to consensus pt)
        if v < tol:
            break

    f_vals = f(x) # obj vals at last iteration
    best_idx = np.argmin(f_vals)

    return (x[best_idx], best_val_hist[-1], np.array(consensus_hist), np.array(particle_hist), np.array(obj_val_hist), np.array(best_val_hist), np.array(variance_hist))

def naive(f, d, domain=(-32,32), alg=cbo, seed=0):
    _, _, consensus_hist, part_hist, _, _, _ = alg(f, d)
    n_iters = len(consensus_hist)
    n_particles = part_hist.shape[1]
    NFE_comp_cost = n_particles * n_iters

    rng = np.random.default_rng(seed) # set up seeded random num generator
    lo, hi = domain
    x = rng.uniform(lo, hi, size=(NFE_comp_cost, d)) # x = (x_1, x_2, ..., x_n)

    f_vals = f(x)
    best_idx = np.argmin(f_vals) # index of lowest f val

    # Variance history
    mean = x.mean(axis=0) # expectation: average all part points
    v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance

    # return best part pos, best obj val, x positions, obj vals, num of iterations, variance
    return (x[best_idx], f_vals[best_idx], np.array(x), np.array(f_vals), len(f_vals), v)

def naive_uniform(f, d, domain=(-32,32), alg=cbo, seed=0):
    _, _, consensus_hist, part_hist, _, _, _ = alg(f, d)
    n_iters = len(consensus_hist)
    n_particles = part_hist.shape[1]
    NFE_comp_cost = n_particles * n_iters

    # generate uniform grid of points
    lo, hi = domain
    pts_per_dim = round(NFE_comp_cost ** (1/d)) # same num of pts on each axis
    lin = np.linspace(lo, hi, pts_per_dim)
    mesh = np.meshgrid(*([lin] * d))
    x = np.stack([m.ravel() for m in mesh], axis=1) # shape: (pts_per_dim**d, d)

    f_vals = f(x)
    best_idx = np.argmin(f_vals) # index of lowest f val

    # Variance history
    mean = x.mean(axis=0) # expectation: average all part points
    v = 0.5*np.mean(np.sum((x-mean)**2, axis=1)) # variance

    # return best part pos, best obj val, num of iterations
    return (x[best_idx], f_vals[best_idx], np.array(x), np.array(f_vals), len(f_vals), v)

# Algorithm 1: ZO-AdaMM (Section 3 Chen paper) (FOR 1 PARTICLE)
def zo_adamm(grad_fn, x0, T=1000, alpha=0.01, beta1=0.9, beta2=0.999, v0=1e-5, lo=None, hi=None, return_hist=False):
    x = x0.copy() # optimize on parameter vector x
    m = np.zeros_like(x) # momentum
    v = np.zeros_like(x) # avg of squared grads
    vhat = np.full_like(x, v0) # Chen paper initializes v0 = vhat0 = 1e-5

    x_hist = [x.copy()]

    # directly from section 3 of Chen paper Algorithm 1
    for t in range(1, T + 1):
        g = grad_fn(x) # FO gradient, or ZO estimate (below)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        vhat = np.maximum(vhat, v) # (the AMSGrad fix: vhat never decreases)

        # update step
            # coords w historically large grads get smaller steps, and vice versa
            # decaying learning rate
        step = (alpha / np.sqrt(t)) * m / np.sqrt(vhat) # alpha_t = alpha/sqrt(t)
        x = x - step

        if lo is not None: # box constraint
            x = np.clip(x, lo, hi)

        x_hist.append(x.copy())

    # optimized point, point history
    return (x, np.array(x_hist)) if return_hist else x

# gradient estimation for zeroth order AdaMM: noisy grad est build on fn vals
def zo_grad_est(f, x, mu=None, q=10, rng=np.random.default_rng()):
    d = x.size
    mu = mu if mu is not None else 1.0 / np.sqrt(d) # paper's mu ~ 1/sqrt(T d)
    fx = f(x)
    g = np.zeros_like(x)
    for _ in range(q): # over q random directions
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u) # draw rand direction uniform on the unit sphere
        # equation 1 from the paper
        g += (d / mu) * (f(x + mu * u) - fx) * u
    return g / q # return single avg grad over q random directions

def zo_adamm_batch(f, d, domain=(-32,32), n_particles=25, q=10, mu=1e-3, alpha=1.0, beta1=0.9, beta2=0.3, v0=1e-5, iters=1000, seed=0): # num of iters T
    rng = np.random.default_rng(seed)
    lo, hi = domain
    x = rng.uniform(lo, hi, size=(n_particles, d))

    m = np.zeros_like(x)
    v = np.zeros_like(x)
    vhat = np.full_like(x, v0)

    particle_hist = [x.copy()]
    obj_val_hist = []
    best_val_hist = []
    running_best = np.inf

    for t in range(1, iters+1): # run over 1000 iters
        f_vals = f(x)
        obj_val_hist.append(f_vals.min())
        running_best = min(running_best, f_vals.min())
        best_val_hist.append(running_best)

        g = zo_grad_est_batch(f, x, f_vals, mu=mu, q=q, rng=rng)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        vhat = np.maximum(vhat, v) # (the AMSGrad fix: vhat never decreases)

        x = x - (alpha / np.sqrt(t)) * m / np.sqrt(vhat)
        x = np.clip(x, lo, hi)

        particle_hist.append(x.copy())

    f_vals = f(x)
    best_idx = np.argmin(f_vals)
    running_best = min(running_best, f_vals.min())
    # best x pos, running best obj val, part hist, ending f vals, obj val hist, running best hist
    return (x[best_idx], running_best, np.array(particle_hist), f_vals, np.array(obj_val_hist), np.array(best_val_hist))
    # for later cbo comparison, total NFE = n_particles * iters * (q+1) evaluations


# gradient estimation for zeroth order AdaMM (batch of random pts): noisy grad est build on fn vals
def zo_grad_est_batch(f, x, fx, mu=1e-3, q=10, rng=None):
    # takes all particles, returns one grad est per particle
    rng = rng or np.random.default_rng()
    N, d = x.shape
    g = np.zeros_like(x)
    for _ in range(q):  # over q random directions
        u = rng.standard_normal((N,d)) # only difference
        u /= np.linalg.norm(u, axis=1, keepdims=True) # draw rand direction uniform on the unit sphere
        # equation 1 from the paper
        g += (d / mu) * (f(x + mu * u) - fx)[:, None] * u
    return g / q # return single avg grad over q random directions


if __name__ == "__main__": # runs when this is the main file of this execution
    print("BASE CASE CBO ON ACKLEY")

    best_x, running_best, consensus_hist, part_hist, obj_val_hist, best_val_hist, var_hist = cbo(ackley, d=2, seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
    print(f"Iterations: {len(consensus_hist)}")
    print("True minimum: f(0,0) = 0")

    print("---------------------------------------")

    print("BASE CASE CBO ON RASTRIGIN")

    best_x, running_best, consensus_hist, part_hist, obj_val_hist, best_val_hist, var_hist = cbo(rastrigin, d=2, seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
    print(f"Iterations: {len(consensus_hist)}")
    print("True minimum: f(0,0) = 0")

    print("---------------------------------------")
    print("SGD on ACKLEY")

    best_x, running_best, part_hist, f_vals, obj_val_hist, best_val_hist = sgd(ackley, d=2, seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
    print(f"Iterations: {len(obj_val_hist)}")
    print("True minimum: f(0,0) = 0")

    print("---------------------------------------")
    print("SGD on RASTRIGIN")

    best_x, running_best, part_hist, f_vals, obj_val_hist, best_val_hist = sgd(rastrigin, d=2, domain=(-5.12, 5.12), seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Ending/Best f(x): {obj_val_hist[-1]:.6f}")
    print(f"Iterations: {len(obj_val_hist)}")
    print("True minimum: f(0,0) = 0")

    print("---------------------------------------")
    print("NAIVE ALGORITHM RESULTS")

    best_x_naive, best_val_naive, x_positions_naive, f_vals_naive, n_part_naive, variance = naive(ackley, d=2)

    print(f"Best x position found: {best_x_naive}")
    print(f"Best f(x): {best_val_naive:.6f}")
    print(f"num particles: {len(x_positions_naive)})")
    print(f"Variance: {variance}")
    print("True minimum: f(0,0) = 0")

    print("---------------------------------------")
    print("NAIVE ALGORITHM (UNIFORM DISTR) RESULTS")

    best_x_naive, best_val_naive, x_positions_naive, f_vals_naive, n_part_naive, variance = naive_uniform(ackley, d=2)

    print(f"Best x position found: {best_x_naive}")
    print(f"Best f(x): {best_val_naive:.6f}")
    print(f"num particles: {len(x_positions_naive)})")
    print(f"Variance: {variance}")
    print("True minimum: f(0,0) = 0")

    # print("---------------------------------------")
    # print("ZO-ADAMM (SINGLE PARTICLE) ON ACKLEY")

    # def single(f):
    #     return lambda x: float(f(x[None, :])[0])

    # # calculation of exact gradient
    # def fo_grad(f):
    #     def g(x):
    #         xt = torch.tensor(x[None, :], dtype=torch.float64, requires_grad=True)
    #         f(xt).sum().backward()
    #         return xt.grad.numpy()[0]
    #     return g

    # f1 = single(ackley)
    # x0 = np.array([20.0, 25.0]) # starting point
    # T = 1000
    # q = 10
    # zo_rng = np.random.default_rng(1) # seed

    # # first-order AdaMM --> exact gradient of ackley
    # x_fo, hist_fo = zo_adamm(fo_grad(ackley), x0, T=T, alpha=1.0, beta1=0.9, beta2=0.999, lo=-32, hi=32, return_hist=True)

    # # zeroth-order AdaMM
    # x_zo, hist_zo = zo_adamm(lambda x: zo_grad_est(f1, x, mu=1e-3, q=10, rng=zo_rng), x0, T=T, alpha=1.0, beta1=0.9, beta2=0.3, lo=-32, hi=32, return_hist=True)

    # print(f"Start: {x0}, f = {f1(x0):.4f}")
    # print(f"FO AdaMM final x: {x_fo}, f = {f1(x_fo):.6f}")
    # print(f"ZO-AdaMM final x: {x_zo}, f = {f1(x_zo):.6f}")
    # print(f"Path shape (FO):  {hist_fo.shape}")   # (T+1, 2)
    # print("True minimum: f(0,0) = 0")

    print("---------------------------------------")
    print("ZO-ADAMM (PARTICLES) ON ACKLEY")

    N, iters, q = 25, 1000, 10
    best_x, running_best, part_hist, f_end, obj_val_hist, best_val_hist = zo_adamm_batch(ackley, d=2, n_particles=N, iters=iters, q=q, alpha=1.0, seed=0)

    print(f"Best x position found: {best_x}")
    print(f"Running Best f(x): {running_best:.6f}")
    print(f"Particles ending with f < 0.1: {(f_end < 0.1).sum()}/{N}")
    print(f"Median ending f(x): {np.median(f_end):.4f}")
    print(f"particle_hist shape: {part_hist.shape}") # (iters+1, N, 2)
    print(f"Function evaluations: {N * iters * (q + 1) + N}")
    print("True minimum: f(0,0) = 0")
