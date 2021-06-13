import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims

np.random.seed(1337)

# 1. - Create 1D noisy linear data samples
a = 0.5
b = 4
N = 50
x = np.random.uniform(0, 10, size=N)
t = a * x + b + np.random.uniform(-0.8, 0.8, size=x.shape)

X = np.concatenate(
    (
        np.expand_dims(x, axis=1),
        np.expand_dims(t, axis=1)
    ),
    axis=1
)

# 2. - Linear unit
def linear_unit(w, x):
    return np.dot(x, w)

# 2. - Loss function (MSE)
def mse(y, t):
    return np.square(y-t).mean()

# 3. - Compute the gradient of the loss
def grad_mse(y, t, x):
    return (y - t) * x

# 4. Intitialize weights randomly and choose a learning rate
w_init = np.array([1, -1])
eta = 0.005

def gradient_descent(w, eta, adaptive_lr=False, max_iter=1000):
    """
    Linear regression via Gradient Descent.

    Args:
        w (np.array): 1D array of weights (including w_0 = bias)
        eta (float): learning rate for gradient descent optimization
        adaptive_lr (bool): whether to use an adaptive learning rate
        max_iter (int): maximum number of iterations
    Returns:
        w: the optimized weights
        i: number of required epochs
    """
    prev_loss, loss_delta = None, None
    i = 0
    while True:
        np.random.shuffle(X)
        grad, loss = 0, 0
        for sample in X:
            # Create x_bias and target t
            x_bias = np.concatenate(
                ([1], sample[:-1]),
                axis=0
            )
            t = sample[-1]

            # Predict
            y = linear_unit(w, x_bias)
            # Compute loss + save gradient
            loss += mse(y, t)
            grad += grad_mse(y, t, x_bias)

        # Divide gradient by number of samples
        grad = 2*np.divide(grad, N)

        # Update weights
        w = w - eta * grad

         # Adaptive learning rate
        if prev_loss is not None and adaptive_lr:
            loss_delta = np.abs(loss - prev_loss)
            if loss_delta < 0:
                eta = 1.1*eta
            else:
                eta = 0.5*eta

        # Stopping criteria
        i += 1
        if loss_delta is not None and loss_delta < 1e-5:
            print("Stopping because loss isn't changing...")
            break

        if i >= max_iter:
            print("Max iterations reached. Stopping...")
            break

        if np.linalg.norm(grad) < 1e-4:
            print("Stopping cuz gradient is too small.")
            break

        prev_loss = loss

    return w, i


# GD
##
new_w, epochs = gradient_descent(w_init, eta)
adaptive_w, adaptive_epochs = gradient_descent(w_init, eta, adaptive_lr=True)
print(f"Epochs: {epochs} vs adaptive epochs: {adaptive_epochs}")

# Plot
##
plt.clf()
plt.scatter(x, t, marker="x")

# Plot init w
x = np.linspace(0, 10, 4)
x_bias = np.concatenate(
    (
        np.ones((x.shape[0], 1)),
        np.expand_dims(x, axis=1)
    ),
    axis=1
)
y_line = linear_unit(np.array([b, a]), x_bias)
plt.plot(x, y_line, "k--", label="Original line")

# Plot regression line
y_w = linear_unit(new_w, x_bias)
plt.plot(x, y_w, "r", label=f"Regressed line e={epochs}")

# Plot initial weight line
y_init = linear_unit(w_init, x_bias)
plt.plot(x, y_init, "r--", label="Initial w line")

plt.legend()
plt.savefig("linear_regression.png")