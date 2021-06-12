import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal


np.random.seed(1337)
N = 200

## 1. - Generate two separable normal distributed datasets in 2D
# covariance matrix which ensures that the dataset is separable
cov = np.array([[0.02, 0], [0, 1]])

instances_A = np.random.multivariate_normal([4, 0], cov, N)
instances_B = np.random.multivariate_normal([5, 1], cov, N)

## 2. - Label one dataset with 1 and other with -1
instances_A = np.column_stack((instances_A, np.ones(N)))
instances_B = np.column_stack((instances_B, -np.ones(N)))

## 3. - Shuffle both datasets together
dataset = np.vstack((instances_A, instances_B))
np.random.shuffle(dataset)

## 4. - Implement perceptron with numpy matrices
D = 3
O = 1

## 5. - Initialize the weights randomly
w = np.random.uniform(low=-1, high=1, size=D)

## 6. Define an appropriate stopping criterion
MAX_ITER = 500
iteration = 0
while True:
    weights_updated = False

    for i, sample in enumerate(dataset):
        # Split sample into input / target
        t = sample[-1]
        X = np.concatenate(
            ([1], sample[:-1])
        )

        # Predict
        y = np.dot(w, X)

        ## 7. - Apply perceptron learning rule
        # If wrongly classified
        if y * t < 0:
            # Update weights
            w = w + t * X
            weights_updated = True

    # Stopping criterion 1.
    if not weights_updated:
        print("Finished after {} iterations.".format(iteration))
        break

    # Stopping criterion 2.
    iteration += 1
    if iteration >= MAX_ITER:
        print("Maximum iterations reached. Stopping...")
        break

print(f"y = {w[0]:.2f} + {w[1]:.2f}*x_1 + {w[2]:.2f}*x_2")


## 8. - Plot the sample and learned decision boundary
# Points
plt.clf()
plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])

# Line
bias = w[0]
c = -bias / w[2]
m = -w[1] / w[2]
line_x = np.linspace(-10, 10, 2)
line_y = line_x * m + c
plt.plot(line_x, line_y, "--k")
plt.xlim([3, 6])
plt.ylim([-4, 4])

plt.savefig("test.png")
