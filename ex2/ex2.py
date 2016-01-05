import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def plot_data(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos, 0].flat, X[pos, 1].flat, 'k+', markersize=7)
    plt.plot(X[neg, 0].flat, X[neg, 1].flat, 'yo', markersize=7)
    plt.legend(["Admitted", "Not admitted"], numpoints=1)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")

def sigmoid(z):
    return 1 / ( 1 + np.power(np.e, -z))

def cost_function(theta, X, y):
    theta = np.matrix(theta)
    [m, n] = X.shape
    cost = (1 / m ) * (-y.T * np.log(sigmoid(X * theta.T)) - ( 1 - y ).T * np.log( 1 - sigmoid(X * theta.T)))
    return cost.A1

def gradient_function(theta, X, y):
    theta = np.matrix(theta)
    [m, n] = X.shape
    #import pdb; pdb.set_trace()
    gradient = ((sigmoid(X * theta.T) - y).T * X).T / m;
    return gradient.T.tolist()[0]

data = np.matrix(np.loadtxt('ex2data1.txt', delimiter=','))

X = data[:, 0:2]
y = data[:, 2]

plot_data(X, y)
plt.show()

[m, n] = X.shape

# setup data matrix; add ones for the intercept terms
X = np.hstack([np.ones([m, 1]), X])

# initialize fitting parameters
initial_theta = np.zeros([n + 1])

# compute and display initial cost and gradient
cost = cost_function(initial_theta, X, y)
gradient = gradient_function(initial_theta, X, y)

print('cost at initial theta (zeros):\n\t', cost)
print('gradient at initial theta (zeros):\n\t', gradient)

# import pdb; pdb.set_trace()
sol = optimize.fmin(cost_function, x0 = initial_theta, args = (X, y))  #fprime = gradient_function,

print(sol)

