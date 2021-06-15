import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(1337)

# 1. - Load dataset
def load_dataset(filename, n_inputs):
    f = open(filename, "r")
    lines = f.readlines()

    dataset = []
    # Read dataset line by line
    for line in lines:
        # Split line
        sample = list(map(np.float32, line.strip().split(",")))
        dataset.append(sample)

    np.random.shuffle(dataset)
    return np.array(dataset)

banknote_dataset = load_dataset("data/banknote_authentication.txt", n_inputs=4)
X_bank = banknote_dataset[:, :-1]
# Normalize
X_bank = StandardScaler().fit_transform(X_bank)
t_bank = banknote_dataset[:, -1]

email_dataset = load_dataset("data/spambase.data", n_inputs=58)
X_email = email_dataset[:, :-1]
# Normalize
X_email = StandardScaler().fit_transform(X_email)
t_email = email_dataset[:, -1]

# 2. - Implement network
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def add_bias(x):
    return np.pad(x, ((0, 0), (1, 0)), constant_values=1)

def cross_entropy_loss(y, t):
    y = y.squeeze()
    return -np.mean(t * np.log(y + 1e-47) + (1-t) * np.log(1 - y + 1e-47))

def accuracy(y, t):
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    y = y.squeeze()
    return (y == t).mean()

def forward(X, W1, W2):
    a = np.dot(X, W1.T)
    h_ = sigmoid(a)
    h = add_bias(h_)
    # Logits
    z = np.dot(h, W2.T)
    y = sigmoid(z)
    return y, h

# 3. - Gradient descent
def gradient_W1(Y, T, H, W2, X):
    left = (Y - np.expand_dims(T, axis=1)) * W2 * H * (1 - H)
    return np.dot(left.T, X)

def gradient_W2(Y, T, H):
    N = Y.shape[0]
    return 2 * np.dot((Y - np.expand_dims(T, axis=1)).T, H) / N

def descent(X, T, W1, W2, eta, max_iter=1000):
    losses = []
    accuracies = []
    i = 0
    X_bias = add_bias(X)

    while True:
        Y, H = forward(X_bias, W1, W2)

        loss = cross_entropy_loss(Y, T)
        acc = accuracy(Y, T)
        grad_W1 = gradient_W1(Y, T, H, W2, X_bias)
        grad_W1 = grad_W1[1:, :]
        grad_W2 = gradient_W2(Y, T, H)

        assert grad_W1.shape == W1.shape, f"{grad_W1.shape} != {W1.shape}"
        assert grad_W2.shape == W2.shape, f"{grad_W2.shape} != {W2.shape}"

        W1 = W1 - eta * grad_W1
        W2 = W2 - eta * grad_W2

        losses.append(np.squeeze(loss))
        accuracies.append(acc)
        i += 1
        if i >= max_iter:
            print("Max iterations reached. Stopping...")
            break

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

K = 100
W1_bank = np.random.uniform(size=(K, X_bank.shape[1] + 1))
W2_bank = np.random.uniform(size=(1, K+1))

bank_losses, bank_acc, _, _ = descent(X_bank, t_bank, W1_bank, W2_bank, 0.1)
plot_result(
    "banknote_result.png",
    bank_losses,
    bank_acc,
    title="Banknote Forgery training"
)

W1_email = np.random.uniform(size=(K, X_email.shape[1] + 1))
W2_email = np.random.uniform(size=(1, K+1))
email_losses, email_acc, _, _ = descent(X_email, t_email, W1_email, W2_email, 0.01)
plot_result(
    "email_result.png",
    email_losses,
    email_acc,
    title="Spam Classification"
)
