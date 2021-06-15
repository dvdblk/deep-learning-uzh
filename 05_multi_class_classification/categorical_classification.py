import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris, load_digits

rng = np.random.default_rng(1337)

# 1. - Load dataset
def add_bias(x):
    return np.pad(x, ((0, 0), (1, 0)), constant_values=1)

def get_iris():
    iris_data = load_iris()

    combined = np.hstack(
        (iris_data.data, np.expand_dims(iris_data.target, axis=1))
    )
    np.random.shuffle(combined)

    # (150, 5)
    X_iris = add_bias(combined[:, 0:4])
    # (150,)
    t_iris = combined[:, -1]
    # (150, 3)
    t_iris_oh = OneHotEncoder(
        categories="auto"
    ).fit_transform(t_iris.reshape(-1, 1)).toarray()

    return X_iris, t_iris_oh

def get_mnist():
    mnist_data = load_digits()

    combined = np.hstack(
        (mnist_data.data, np.expand_dims(mnist_data.target, axis=1))
    )
    np.random.shuffle(combined)

    X_mnist = add_bias(combined[:, :64])
    t_mnist = combined[:, -1]
    t_mnist_oh = OneHotEncoder(
        categories="auto"
    ).fit_transform(t_mnist.reshape(-1, 1)).toarray()

    return X_mnist, t_mnist_oh


def batch(X, T, B=16):
    """
    Create a batch from input data

    Args:
        X (np.array): the input data (N, D)
        T (np.array): targets (D,)
        B (int): batch_size
    """
    # Shuffle
    dataset = np.concatenate(
        (
            X,
            T
        ),
        axis=1
    )
    rng.shuffle(dataset)

    N = X.shape[0]
    assert N >= B
    n_batches = N // B
    for i in range(n_batches+1):
        yield dataset[i*B:(i+1)*B, :]

# 2. - Implement network
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    # Get largest logit in each training instance
    largest_logit = np.expand_dims(np.max(z, axis=1), axis=1)
    denominator = np.exp(z - largest_logit).sum()
    res = np.exp(z - largest_logit) / denominator
    assert np.isclose(res.sum(), 1), f"{res.sum()}"
    return res

def forward(X, W1, W2):
    a = np.dot(X, W1.T)
    h_ = sigmoid(a)
    h = add_bias(h_)

    z = np.dot(h, W2.T)
    y = softmax(z)

    return y, h

# 3. - Compute accuracy
def accuracy(Y, T, W1, W2):
    return (np.argmax(Y, axis=1) == np.argmax(T, axis=1)).mean()

# 4. - Gradient descent step
def cross_entropy_loss(y, t):
    N = y.shape[0]
    return -(t * np.log(y + 1e-47)).sum(axis=(1, 0)) / N

def gradient_W1(y, t, h, w, x):
    N = y.shape[0]
    left = np.dot((y-t), w) * h * (1-h)
    return 2 * np.dot(left.T, x) / N

def gradient_W2(y, t, h):
    N = y.shape[0]
    return 2 * np.dot((y - t).T, h) / N

def descent(X, T, W1, W2, eta, B=16, max_iter=10000):
    losses, accuracies = [], []
    for e in range(max_iter):
        batch_loss = 0
        batch_acc = 0
        for samples in batch(X, T, B):
            x = samples[:, :X.shape[1]]
            t = samples[:, -T.shape[1]:]

            y, h = forward(x, W1, W2)
            batch_loss += cross_entropy_loss(y, t)
            batch_acc += accuracy(y, t, W1, W2)

            grad_W1 = gradient_W1(y, t, h, W2, x)
            grad_W1 = grad_W1[1:, :]
            grad_W2 = gradient_W2(y, t, h)

            W1 = W1 - eta * grad_W1
            W2 = W2 - eta * grad_W2

        if e % 10 == 0:
            n_batches = np.ceil(X.shape[0] / B)
            losses.append(batch_loss / n_batches)
            accuracies.append(batch_acc / n_batches)

    print(f"Max iterations reached. Stopping at epoch #{e}...")
    return losses, accuracies, W1, W2


def plot_result(filename, losses, accuracies, title):
    plt.clf()
    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_xscale("log")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(np.arange(len(losses)), losses, label="Loss", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:red"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(np.arange(len(accuracies)), accuracies, label="Accuracy", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.suptitle(title)
    plt.savefig(filename)


# Iris
X_iris, t_iris = get_iris()
K = 950
W1 = rng.normal(loc=0, scale=1, size=(K, X_iris.shape[1]))
W2 = rng.normal(loc=0, scale=1, size=(t_iris.shape[1], K+1))
losses, acc, _, _ = descent(X_iris, t_iris, W1, W2, 0.0001, max_iter=1000)

plot_result("iris_training.png", losses, acc, "Iris Dataset")
print(f"Iris accuracy {acc[-1]}")


# MNIST
X_mnist, t_mnist = get_mnist()
K = 1000
W1 = rng.normal(loc=0, scale=1, size=(K, X_iris.shape[1]))
W2 = rng.normal(loc=0, scale=1, size=(t_iris.shape[1], K+1))
losses, acc, _, _ = descent(X_iris, t_iris, W1, W2, 0.0001, max_iter=1000)

plot_result("mnist_training.png", losses, acc, "MNIST Dataset")
print(f"MNIST accuracy {acc[-1]}")