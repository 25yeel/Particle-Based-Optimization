import torch
import torch.nn.functional as F
from torchvision import datasets

# Same function as mnist_data.py
# DOWNLOAD MNIST, (but LeNet-5 expects 32x32x1 input)

N_TRAIN = 10_000

def load_lenet_mnist(n_train=N_TRAIN, seed=0, root="./mnist_data"):
    """ X_* : (n, 1, 32, 32) float tensor, pixel values scaled to [0,1]
        y_* : (n, 10) one-hot float tensor
        Downloads MNIST via torchvision on first call (cached under `root` after that). """
    train_raw = datasets.MNIST(root=root, train=True, download=True)
    test_raw = datasets.MNIST(root=root, train=False, download=True)

    def to_tensors(raw, n):
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(raw), generator=rng)[:n] # random subset of size n

        imgs = raw.data[idx].float() / 255.0 # (n, 28, 28)
        imgs = imgs.unsqueeze(1) # (n, 1, 28, 28) add channel dim
        imgs = F.pad(imgs, (2, 2, 2, 2)) # -> (n, 1, 32, 32), matching LeNet-5's original input size
        y = torch.nn.functional.one_hot(raw.targets[idx], num_classes=10).float() # (n, 10)
        return imgs, y

    X_train, y_train = to_tensors(train_raw, n_train)
    X_test, y_test = to_tensors(test_raw, len(test_raw)) # use the full 10k test set

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_lenet_mnist()
    # confirm: X --> (10000, 1, 32, 32), Y --> (10000, 10)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"example label (one-hot): {y_train[0]}")
    print(f"digit distribution in y_train: {y_train.sum(dim=0)}")
