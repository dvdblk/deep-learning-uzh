import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rng = np.random.default_rng(1)


# 1. - Batch processing
def batch(X, T, B=32):
    """
    Create a batch from input data

    Args:
        X (np.array): the input data (N, D)
        T (np.array): targets (D,)
        B (int): batch_size
    """
    # Add bias
    X_bias = np.pad(
        X,
        ((0, 0), (1, 0)),
        constant_values=1
    )
    # Shuffle
    dataset = np.concatenate(
        (
            X_bias,
            T
        ),
        axis=1
    )
    rng.shuffle(dataset)

    N = X_bias.shape[0]
    assert N >= B
    n_batches = N // B
    for i in range(n_batches+1):
        yield dataset[i*B:(i+1)*B, :]


# 2. - Multi-target network
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W1, W2):
    """
    Args:
        X: (N, D+1)
        W1: (K, D+1)
        W2: (O, K)
    """
    # (N, K)
    A = np.dot(X, W1.T)
    H_ = sigmoid(A)
    H = np.pad(H_, ((0, 0), (1, 0)), constant_values=1)

    Y = np.dot(H, W2.T)
    return Y, H

# 3. - Gradient Descent Step
def loss_frobenius(y, t):
    N = y.shape[0]
    #print(y.shape, t.shape)
    return np.linalg.norm(y - t, ord="fro") / N

def gradient_W1(y, t, h, w, x):
    N = y.shape[0]
    left = np.dot((y-t), w) * h * (1-h)
    return 2 * np.dot(left.T, x) / N

def gradient_W2(y, t, h):
    N = y.shape[0]
    return 2 * np.dot((y - t).T, h) / N

def descent(X, T, W1, W2, eta, mu=0, prev_W1=None, prev_W2=None):
    """
    Args:
        X: (N, D+1)
        T: (N, 1)
        W1: (K, D+1)
        W2: (O, K)
        eta: learning rate
        mu: momentum term (default=0)
    """
    Y, H = forward(X, W1, W2)
    assert Y.shape[0] == X.shape[0]

    loss = loss_frobenius(Y, T)
    grad_W1 = gradient_W1(Y, T, H, W2, X)
    grad_W1 = grad_W1[1:, :]
    grad_W2 = gradient_W2(Y, T, H)

    assert grad_W1.shape == W1.shape and grad_W2.shape == W2.shape

    # Momentum
    if prev_W1 is not None and prev_W2 is not None:
        W1 = W1 - eta * grad_W1 + mu * (W1 - prev_W1)
        W2 = W2 - eta * grad_W2 + mu * (W2 - prev_W2)
    else:
        W1 = W1 - eta * grad_W1
        W2 = W2 - eta * grad_W2

    return W1, W2, loss


# 4. - Data Set Loading
def get_data(filename):
    student = pd.read_csv(filename, delimiter=";")
    student_orig = student.copy()

    for col in range(student.shape[1]):
        unique_vals = student.iloc[:, col].unique().tolist()
        # If binary
        if len(unique_vals) == 2:
            # Replace with -1/1
            student.iloc[:, col].replace(unique_vals, [-1, 1], inplace=True)

    X_1 = student.iloc[:, 0:8].to_numpy(dtype=np.float32)
    X_2 = student.iloc[:, 13:31].to_numpy(dtype=np.float32)
    X = np.concatenate(
        (X_1, X_2),
        axis=1
    )
    X = np.hstack((X_1, X_2))
    T = student.iloc[:, -3:].to_numpy(dtype=np.float32)

    return X, T, student_orig

X, T, df_mat = get_data("data/student-mat.csv")
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


# 5. - Learning
D = X.shape[1]          # Number of features
B = 32                  # Batch size
K = 150                 # Hidden layer size
O = T.shape[1]          # Number of outputs
eta = 0.01              # Learning rate
max_epochs = 10000      # Maximum number of epochs

W1 = rng.uniform(size=(K, D+1))
W2 = rng.uniform(size=(O, K+1))

def learn(X, T, B, W1, W2, eta, momentum=False):
    mu = 0.9 if momentum else 0
    losses = []
    prev_W1 = None
    prev_W2 = None
    for e in range(max_epochs):
        loss = 0
        for b in batch(X, T, B):
            x = b[:, :-O]
            t = b[:, -O:]

            W1_new, W2_new, err = descent(x, t, W1, W2, eta, mu=mu, prev_W1=prev_W1, prev_W2=prev_W2)
            loss += err

            prev_W1 = W1
            prev_W2 = W2

            W1 = W1_new
            W2 = W2_new

        losses.append(loss)
    return losses, W1, W2

sgd_losses, _, _ = learn(X, T, B, W1, W2, eta, momentum=False)
sgd_momentum_losses, mom_W1, mom_W2 = learn(X, T, B, W1, W2, eta, momentum=True)

plt.clf()
plt.yscale("log")
plt.xscale("log")
plt.plot(np.arange(len(sgd_losses)), sgd_losses, label="Stochastic Gradient Descent")
plt.plot(np.arange(len(sgd_momentum_losses)), sgd_momentum_losses, label="Stochastic Gradient Descent + Momentum")
plt.suptitle("Losses for training methods")
plt.legend()
plt.savefig("loss.png")
plt.clf()

# 7. - Evaluation
# Select students by parameter
female_students = X[(df_mat["sex"] == "F").to_numpy()]
male_students = X[(df_mat["sex"] == "M").to_numpy()]
paid_classes_yes = X[(df_mat["paid"] == "yes").to_numpy()]
paid_classes_no = X[(df_mat["paid"] == "no").to_numpy()]
romantic_yes = X[(df_mat["romantic"] == "yes").to_numpy()]
romantic_no = X[(df_mat["romantic"] == "no").to_numpy()]

def add_bias(x):
    return np.pad(x, ((0, 0), (1, 0)), constant_values=1)

# Compute estimated average grades from network output
female_y, _ = forward(add_bias(female_students), mom_W1, mom_W2)
male_y, _ = forward(add_bias(male_students), mom_W1, mom_W2)
paid_y, _ = forward(add_bias(paid_classes_yes), mom_W1, mom_W2)
paid_no_y, _ = forward(add_bias(paid_classes_no), mom_W1, mom_W2)
romantic_y, _ = forward(add_bias(romantic_yes), mom_W1, mom_W2)
romantic_no_y, _ = forward(add_bias(romantic_no), mom_W1, mom_W2)

# Print result
result = np.vstack(
    (
        np.mean(female_y, axis=0),
        np.mean(male_y, axis=0),
        np.mean(paid_y, axis=0),
        np.mean(paid_no_y, axis=0),
        np.mean(romantic_y, axis=0),
        np.mean(romantic_no_y, axis=0)
    )
)
print(result.T)