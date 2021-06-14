import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims

np.random.seed(1336)

def mse(y, t):
    return np.square(y-t).mean()

def gradient_1st(y, t, h, w, x):
    N = y.shape[0]
    left = np.dot(np.expand_dims(y-t, axis=1), np.expand_dims(w, axis=0)) * h * (1-h)
    return 2 * np.dot(left.T, x) / N

def gradient_2nd(y, t, h):
    N = y.shape[0]
    return 2 * np.dot(y - t, h) / N

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W, w):
    # (10, 2) x (2, K) = (10, K)
    a = np.dot(X, W.T)
    # (10, K)
    h_ = sigmoid(a)
    # (10, K+1)
    h = np.concatenate(
        (
            np.ones((h_.shape[0], 1)),
            h_,
        ),
        axis=1
    )
    # (1, K+1) x (K+1, 10) = (1, 10)
    y = np.dot(
        np.expand_dims(w, axis=0),
        h.T
    )
    return np.squeeze(y), h

def train_network(X, t, eta=0.01, K=20, max_iter=1000):

    W = np.random.uniform(size=(K, 2))
    w = np.random.uniform(size=K+1)
    X = np.expand_dims(X, axis=1)
    X_bias = np.concatenate(
        (
            np.ones_like(X),
            X
        ),
        axis=1
    )
    i = 0
    losses = []
    while True:

        y, h = forward(X_bias, W, w)
        loss = mse(y, t)
        grad_W = gradient_1st(y, t, h, w, X_bias)
        grad_w = gradient_2nd(y, t, h)


        grad_W = grad_W[1:, :]
        w = w - eta * grad_w
        W = W - eta * grad_W
        i += 1
        losses.append(loss)

        if i >= max_iter:
            print("Max iterations reached. Stopping...")
            break
        if np.linalg.norm(grad_w) < 1e-8 or np.linalg.norm(grad_W) < 1e-8:
            print(f"Gradient close to 0. Stopping... {i}")
            break

        if len(losses) > 2 and np.abs(loss - losses[i-2]) < 1e-8 and np.abs(loss - losses[i-3]) < 1e-8:
            print("Loss hasn't changed. Stopping...")
            break

    return w, W, i, losses

N_samples = 100

# Polynomial
X_poly = np.random.uniform(low=-4.5, high=3.5, size=N_samples)
t_poly = (X_poly**5 + 3*X_poly**4 - 11*X_poly**3 + 10*X_poly + 64) / 100

X_poly_bias = np.concatenate(
    (
        np.ones((X_poly.shape[0], 1)),
        np.expand_dims(X_poly, axis=1)
    ),
    axis=1
)
poly_w, poly_W, poly_e, poly_loss = train_network(X_poly, t_poly, eta=0.1, max_iter=40000, K=50)

plt.plot(np.arange(len(poly_loss)), poly_loss, label="Loss")
plt.ylim(0, 2)
plt.savefig("poly_loss.png")
plt.clf()
plt.scatter(X_poly, t_poly, c="r", marker="x")
y_poly, _ = forward(X_poly_bias, poly_W, poly_w)
order = np.argsort(X_poly)
plt.plot(X_poly[order], y_poly[order], c="g")
plt.savefig("poly_approximation.png")
plt.clf()

# Cos
X_cos = np.random.uniform(low=-2, high=2, size=N_samples)
t_cos = (np.cos(3*X_cos) + 1) / 2

X_cos_bias = np.concatenate(
    (
        np.ones((X_cos.shape[0], 1)),
        np.expand_dims(X_cos, axis=1)
    ),
    axis=1
)
cos_w, cos_W, cos_e, cos_loss = train_network(X_cos, t_cos, eta=0.02, max_iter=200000, K=10)

plt.plot(np.arange(len(cos_loss)), cos_loss, label="Loss")
plt.ylim(0, 1)
plt.savefig("cos_loss.png")
plt.clf()
plt.scatter(X_cos, t_cos, c="r", marker="x")
y_cos, _ = forward(X_cos_bias, cos_W, cos_w)
order = np.argsort(X_cos)
plt.plot(X_cos[order], y_cos[order], c="g")
plt.savefig("cos_approximation.png")
plt.clf()


# Gaussian
X_gauss = X_cos.copy()
t_gauss = np.exp(-1/4 * X_gauss**2)

X_gauss_bias = np.concatenate(
    (
        np.ones((X_gauss.shape[0], 1)),
        np.expand_dims(X_gauss, axis=1)
    ),
    axis=1
)
gauss_w, gauss_W, gauss_e, gauss_loss = train_network(X_gauss, t_gauss, eta=0.01, max_iter=30000, K=20)

plt.plot(np.arange(len(gauss_loss)), gauss_loss, label="Loss")
plt.ylim(0, 1)
plt.savefig("gauss_loss.png")
plt.clf()
plt.scatter(X_gauss, t_gauss, c="r", marker="x")
y_gauss, _ = forward(X_gauss_bias, gauss_W, gauss_w)
plt.plot(X_gauss[order], y_gauss[order], c="g")
plt.savefig("gauss_approximation.png")
plt.clf()