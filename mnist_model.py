import torch

# constants
D_IMG = 784 # flattened 28x28 pixel input dimension
N_CLASSES = 10 # digits 0-9
D_PARAMS = N_CLASSES * D_IMG + N_CLASSES # size of one particle's flattened (theta, B): 7850

def unflatten_params(x):
    # x: (N, D_PARAMS) one vector row per particle --> reshape

    n_particles = x.shape[0]
    theta = x[:, :N_CLASSES * D_IMG].reshape(n_particles, N_CLASSES, D_IMG)
    b = x[:, N_CLASSES * D_IMG:]

    # returns theta: (N, 10, 784) = (particles, classes, pixels), b: (N, 10) = (particles, classes)
    return theta, b

def mnist_forward(x, X_batch): # x: (N, D_PARAMS) particle params; X_batch: (m, 784) flattened images
    theta, b = unflatten_params(x)

    # f(x,x̂) in 4.3 --> but batched over particles and images simultaneously
    logits = torch.einsum('nkd,md->nmk', theta, X_batch) + b[:, None, :]
    # (N, m, k) = (particles, images, classes)

    # returns probs: (N, m, 10) per particle per image
    return torch.softmax(torch.relu(logits), dim=-1)

def mnist_loss(x, X_batch, y_batch): # 4.4: L(x), ℓ(f,y)
    # x: (N, D_PARAMS); X_batch: (m, 784); y_batch: (m, 10)

    probs = mnist_forward(x, X_batch) # (N, m, 10)
    probs = probs.clamp(min=1e-12) # avoid log(0)
    ce = -(y_batch[None, :, :] * probs.log()).sum(dim=-1) # (N, m) per-particle, per-sample loss

    return ce.mean(dim=1) # (N,) averaged over the batch --> L_hat(x) per particle

def make_batch_loss(X_data, y_data):
    """ Binds mnist_loss to a fixed dataset (train or test), giving the f(x, batch_idx)
        signature expected by sgd/cbo's mini-batch machinery. """
    def f(x, batch_idx):
        return mnist_loss(x, X_data[batch_idx], y_data[batch_idx])
    return f

if __name__ == "__main__":
    from mnist_data import load_mnist

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
