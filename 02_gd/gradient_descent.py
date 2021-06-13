import numpy as np
import matplotlib.pyplot as plt


def loss(w1, w2):
    return w1**2 + w2**2 + 30 * np.sin(w1) * np.sin(w2)

# 1. - Compute the gradient for the loss fn
def loss_grad(w1, w2):
    partial_w1 = 2*w1 + 30 * np.cos(w1) * np.sin(w2)
    partial_w2 = 2*w2 + 30 * np.sin(w1) * np.cos(w2)
    return np.array([partial_w1, partial_w2])

# 2. - Implement GD for initial w
w = [-0.5, 0.9]

def gradient_descent(w, eta=0.3, max_iter=500):
    """Return the optimized weight vector"""
    iteration = 0
    epsilon = 1e-5
    while True:
        # Get loss for current w
        J_w = loss(w[0], w[1])

        # Compute gradient
        grad_J = loss_grad(w[0], w[1])

        w = w - eta * grad_J

        # Stopping criterion
        iteration += 1
        ## 1: fixed nr of iterations
        if iteration >= max_iter:
            print("Reached max iterations. Stopping...")
            break

        ## 2: little change in loss
        J_w_new = loss(w[0], w[1])
        if np.abs(J_w - J_w_new) < epsilon:
            print("Loss change is small. Stopping...")
            break

        ## 3: gradient is close to 0
        if np.linalg.norm(grad_J) < epsilon:
            print("Gradient is zero. Stopping...")
            break

    print("w: ", w)
    return w

new_w = gradient_descent(w)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 200)

# Make data.
w_1 = np.arange(-10, 10, 0.25)
w_2 = np.arange(-10, 10, 0.25)
w_1, w_2 = np.meshgrid(w_1, w_2)

# Plot the surface.
from matplotlib import cm
surf = ax.plot_surface(
    w_1,
    w_2,
    loss(w_1, w_2),
    cmap=cm.hsv,
    linewidth=0,
    antialiased=False,
    alpha=0.2
)

ax.scatter(
    w[0],
    w[1],
    loss(w[0], w[1]),
    s=100,
    c='k',
    alpha=1
)
ax.scatter(
    new_w[0],
    new_w[1],
    loss(new_w[0], new_w[1]),
    s=150,
    c='r',
    alpha=1
)

plt.savefig("gradient_descent.png")