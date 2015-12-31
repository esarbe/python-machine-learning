import numpy as np
from numpy import zeros, linalg

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import GradientDescent as gd

from data import *

def normal_equations(X, y):
  return linalg.inv(X.T * X) * X.T * y

print("loading data")
data = loadData("ex1data2.txt")

X = data[:, 0:2]
y = data[:, 2]
(m, num_features) = data.shape

print("First 10 examples from the dataset");
for x, y_ in zip(X[:10,:], y):
  print("x = ", x.A1, ", y = ", y_.A1)

print("Normalizing features... ")

# scale features and set them to zero mean
X, mu, sigma = normalizeFeatures(X)

# add intercept term to X
X = np.append(np.ones((m, 1)), X, 1)

# Gradient Descent with multiple variables

alpha = 0.6
num_iters = 20

# initialize theta and run Gradient Descent
theta = zeros((num_features, 1))
theta, J_history = gd.performGradientDescent(X, y, theta, alpha, num_iters)

plt.plot(range(len(J_history)), J_history)
plt.xlabel("Number of iterations")
plt.ylabel("Cost J")
plt.show()

# display Gradient descent's result
print("Theta computed from gradient descent:")
print(theta.A1)

# estimate the price of a 1650 sq-ft, 3br house
# first columns is all-ones, so it doesn't need to be normalized
X1 = np.append(np.matrix([1]), (np.matrix([1650, 3]) - mu) / sigma, 1)
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):")
print("\t$", (X1 * theta).A1)

print("Solving with normal equations")
theta = normal_equations(X, y)

print("Theta computed with normal equations")
print(theta.A1)

print("Predicted price of a 1650 sq-ft, 3 br house (using normal equations):")
print("\t$", (X1 * theta).A1)


