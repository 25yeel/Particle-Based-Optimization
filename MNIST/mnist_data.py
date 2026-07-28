import torch
from torchvision import datasets

# DOWNLOAD MNIST

# (parameters 4.6 in Carrillo): n = 10^4 training points
N_TRAIN = 10_000

def load_mnist(n_train=N_TRAIN, seed=0, root="./mnist_data"):
    """ X_* : (n, 784) float tensor, pixel values scaled to [0,1]
        y_* : (n, 10) one-hot float tensor
        Downloads MNIST via torchvision on first call (cached under `root` after that). """
    train_raw = datasets.MNIST(root=root, train=True, download=True)
    test_raw = datasets.MNIST(root=root, train=False, download=True)

    def flatten_and_onehot(raw, n):
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(raw), generator=rng)[:n] # random subset of size n of all indices (paper subsamples n=10^4 out of 60k train images)

        X = raw.data[idx].float().reshape(len(idx), -1) / 255.0 # (n, 28*28) scaled to [0,1]
        y = torch.nn.functional.one_hot(raw.targets[idx], num_classes=10).float() # (n, 10)
        # return flattened images and one-hot labels
        return X, y

    X_train, y_train = flatten_and_onehot(train_raw, n_train)
    X_test, y_test = flatten_and_onehot(test_raw, len(test_raw)) # use the full 10k test set

    # return tensor arrays to feed to model
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    # confirm: X --> ([10000, 784]) Y --> ([10000, 10])
    # pixel value in [0,1]
    # digit distribution is balanced (can see it's generally 900-1200 per class)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"example label (one-hot): {y_train[0]}")
    print(f"digit distribution in y_train: {y_train.sum(dim=0)}")
