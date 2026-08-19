import torch
import torch.nn.functional as F
from LeNet.lenet_data import load_lenet_mnist

# constants: LeNet-5 layer shapes
N_CLASSES = 10 # digits 0-9
C1_OUT, C1_IN, C1_K = 6, 1, 5   # conv1: 1x32x32 -> 6x28x28
C2_OUT, C2_IN, C2_K = 16, 6, 5  # conv2: 6x14x14 -> 16x10x10
FC1_IN, FC1_OUT = 16 * 5 * 5, 120  # after conv2 + pool: 16x5x5 flattened
FC2_IN, FC2_OUT = 120, 84
FC3_IN, FC3_OUT = 84, N_CLASSES

# (name, shape) for every weight/bias tensor, in flattening order
_PARAM_SHAPES = [
    ("c1_w", (C1_OUT, C1_IN, C1_K, C1_K)), ("c1_b", (C1_OUT,)),
    ("c2_w", (C2_OUT, C2_IN, C2_K, C2_K)), ("c2_b", (C2_OUT,)),
    ("f1_w", (FC1_OUT, FC1_IN)), ("f1_b", (FC1_OUT,)),
    ("f2_w", (FC2_OUT, FC2_IN)), ("f2_b", (FC2_OUT,)),
    ("f3_w", (FC3_OUT, FC3_IN)), ("f3_b", (FC3_OUT,)),
]

def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n

D_PARAMS = sum(_numel(shape) for _, shape in _PARAM_SHAPES) # size of one particle's flattened LeNet-5 weights: 61706
M_BATCH = 50 # m in (4.6): training data mini batch size

# turn flat particle vector back into named weight/bias tensors
def unflatten_params(x):
    # x: (N, D_PARAMS) one vector row per particle --> dict of per-particle tensors

    n_particles = x.shape[0]
    params = {}
    offset = 0
    for name, shape in _PARAM_SHAPES:
        n = _numel(shape)
        params[name] = x[:, offset:offset + n].reshape(n_particles, *shape)
        offset += n

    # each params[name]: (N, *shape) -- one weight/bias tensor per particle
    return params

# run neural network
def lenet_forward(x, X_batch): # x: (N, D_PARAMS) particle params; X_batch: (m, 1, 32, 32) images
    params = unflatten_params(x)
    n_particles = x.shape[0]
    m = X_batch.shape[0]

    # conv weights differ per particle, so F.conv2d can't batch them like the linear
    # model's einsum does -- loop over particles instead (N is small, images are tiny)
    logits = X_batch.new_empty(n_particles, m, N_CLASSES)
    for i in range(n_particles):
        h = F.conv2d(X_batch, params["c1_w"][i], params["c1_b"][i]) # -> (m, 6, 28, 28)
        h = torch.tanh(h)
        h = F.avg_pool2d(h, 2) # -> (m, 6, 14, 14)

        h = F.conv2d(h, params["c2_w"][i], params["c2_b"][i]) # -> (m, 16, 10, 10)
        h = torch.tanh(h)
        h = F.avg_pool2d(h, 2) # -> (m, 16, 5, 5)

        h = h.reshape(m, -1) # -> (m, 400)
        h = torch.tanh(F.linear(h, params["f1_w"][i], params["f1_b"][i])) # -> (m, 120)
        h = torch.tanh(F.linear(h, params["f2_w"][i], params["f2_b"][i])) # -> (m, 84)
        logits[i] = F.linear(h, params["f3_w"][i], params["f3_b"][i]) # -> (m, 10)

    # returns probs: (N, m, 10 class scores (prob distr)) per particle per image
    return torch.softmax(logits, dim=-1)

# score how wrong each particle's network is
def lenet_loss(x, X_batch, y_batch): # x: (N, D_PARAMS); X_batch: (m, 1, 32, 32); y_batch: (m, 10)
    probs = lenet_forward(x, X_batch) # (N, m, 10)
    probs = probs.clamp(min=1e-12) # avoid log(0)
    ce = -(y_batch[None, :, :] * probs.log()).sum(dim=-1) # (N, m) per-particle, per-sample loss

    return ce.mean(dim=1) # (N,) averaged over the batch --> L_hat(x) per particle

# preload dataset (train or test)
def make_batch_loss(X_data, y_data):
    def f(x, batch_idx):
        return lenet_loss(x, X_data[batch_idx], y_data[batch_idx])
    return f

def make_lenet_objective(X_train, y_train, m, seed=0):
    train_loss_fn = make_batch_loss(X_train, y_train) # torch-based f(x, batch_idx)
    gen = torch.Generator().manual_seed(seed) # instantiate ONCE, outside f

    def f(x): # x: numpy (n_particles, D_PARAMS) --> numpy (n_particles,)
        x_t = torch.from_numpy(x).float()
        batch_idx = torch.randperm(len(X_train), generator=gen)[:m]
        with torch.no_grad():
            losses = train_loss_fn(x_t, batch_idx)
        return losses.numpy()

    return f # return explicit data batch instead of torch

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_lenet_mnist()

    torch.manual_seed(0)
    n_particles = 5
    x = torch.randn(n_particles, D_PARAMS, requires_grad=True) # standard normal init

    train_loss_fn = make_batch_loss(X_train, y_train)
    batch_idx = torch.randperm(len(X_train))[:50]

    loss = train_loss_fn(x, batch_idx)
    print("per-particle loss:", loss)
    print("loss shape:", loss.shape) # expect (5,)

    loss.sum().backward() # confirm gradients are finite and shaped (5, D_PARAMS)
    print("grad shape:", x.grad.shape, "grad is finite:", torch.isfinite(x.grad).all().item())

    with torch.no_grad():
        x -= 0.1 * x.grad
    x = x.detach().requires_grad_(True)
    new_loss = train_loss_fn(x, batch_idx)
    print("loss after one GD step:", new_loss)
