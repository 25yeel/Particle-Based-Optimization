# LAST UPDATED 8/19/2026
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from MNIST.mnist_data import load_mnist

# constants
D_IMG = 728 # flattened 26x28 pixel input dimension (top/bottom row cropped, matching paper section 4.3)
N_CLASSES = 10 # digits 0-9
D_PARAMS = N_CLASSES * D_IMG + N_CLASSES # size of one particle's flattened (theta, B): 7850
M_BATCH = 50 # m in (4.6): training data mini batch size
# turn flat particle vector back into weight matrix + bias
def unflatten_params(x):
    # x: (N, D_PARAMS) one vector row per particle --> reshape

    n_particles = x.shape[0]
    theta = x[:, :N_CLASSES * D_IMG].reshape(n_particles, N_CLASSES, D_IMG)
    b = x[:, N_CLASSES * D_IMG:] # remaining 10 entries of 7850

    # returns theta: (N, 10, 728) = (particles, classes, pixels), b: (N, 10) = (particles, classes)
    return theta, b

# run neural network
def mnist_forward(x, X_batch): # x: (N, D_PARAMS) particle params; X_batch: (m, 728) flattened images
    theta, b = unflatten_params(x)

    # f(x,x̂) in 4.3 --> but batched over particles and images simultaneously
    logits = torch.einsum('nkd,md->nmk', theta, X_batch) + b[:, None, :]
    # (N, m, k) = (particles, images, classes)

    # returns probs: (N, m, 10 class scores (prob distr)) per particle per image
    # match paper's ReLu pg. 17
    return torch.softmax(torch.relu(logits), dim=-1)

# score how wrong each particle's network is
def mnist_loss(x, X_batch, y_batch): # 4.4: L(x), ℓ(f,y)
    # x: (N, D_PARAMS); X_batch: (m, 728); y_batch: (m, 10)

    probs = mnist_forward(x, X_batch) # (N, m, 10)
    probs = probs.clamp(min=1e-12) # avoid log(0)
    ce = -(y_batch[None, :, :] * probs.log()).sum(dim=-1) # (N, m) per-particle, per-sample loss

    return ce.mean(dim=1) # (N,) averaged over the batch --> L_hat(x) per particle

# preload dataset (train or test)
def make_batch_loss(X_data, y_data):
    # preload dataset (train or test) so that teh result matches the plain f(x, batch_idx) calling convention sgd/cbo expects
    def f(x, batch_idx):
        return mnist_loss(x, X_data[batch_idx], y_data[batch_idx])
    return f

def make_mnist_objective(X_train, y_train, m, seed=0):
    train_loss_fn = make_batch_loss(X_train, y_train) #torch-based f(x, batch_idx)
    gen = torch.Generator().manual_seed(seed) # instantiate ONCE, outside f

    def f(x): # x: numpy (n_particles, D_PARAMS) --> numpy (n_particles, )
        x_t = torch.from_numpy(x).float()
        batch_idx = torch.randperm(len(X_train), generator=gen)[:m]
        with torch.no_grad():
            losses = train_loss_fn(x_t, batch_idx)
        return losses.numpy()

    return f # return explicit data batch instead of torch

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_mnist()

    torch.manual_seed(0)
    n_particles = 5
    x = torch.randn(n_particles, D_PARAMS, requires_grad=True) # standard normal init, matching the Carrillo paper

    train_loss_fn = make_batch_loss(X_train, y_train)
    batch_idx = torch.randperm(len(X_train))[:50]

    loss = train_loss_fn(x, batch_idx)
    print("per-particle loss:", loss)
    print("loss shape:", loss.shape) # expect (5,)

    loss.sum().backward() # confirm gradients are finite and shaped (5, 7850)
    print("grad shape:", x.grad.shape, "grad is finite:", torch.isfinite(x.grad).all().item())

    # sanity check: a few plain gradient steps should reduce the loss
    with torch.no_grad():
        x -= 0.1 * x.grad
    x = x.detach().requires_grad_(True)
    new_loss = train_loss_fn(x, batch_idx)
    print("loss after one GD step:", new_loss)
