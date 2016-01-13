import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos, 0].flat, X[pos, 1].flat, 'k+', markersize=7)
    plt.plot(X[neg, 0].flat, X[neg, 1].flat, 'yo', markersize=7)
    plt.legend(["x=1", "x=0"], numpoints=1)
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")


def sigmoid(z):
    return 1 / ( 1 + np.power(np.e, -z))


def map_features(X1, X2, degree=6):
    out = np.ones((X1[:,0].shape))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            col = np.multiply(np.power(X1, i-j), np.power(X2, j))
            out = np.hstack([out, col])
    return out


def cost_function(theta, X, y, λ):
    theta = np.matrix(theta)
    [m, n] = X.shape
    cost = (1 / m ) * (-y.T * np.log(sigmoid(X * theta.T)) - ( 1 - y ).T * np.log( 1 - sigmoid(X * theta.T)))
    cost = cost.A1
    reg_cost = cost + (λ / 2 / m) * np.sum(np.power(theta[:,1:], 2))
    return reg_cost


def gradient_function(theta, X, y, λ):
    theta = np.matrix(theta)
    [m, n] = X.shape
    gradient = ((sigmoid(X * theta.T) - y).T * X).T / m;
    theta[0,0] = 0
    reg = (λ / m) * theta
    gradient_reg = gradient + reg.T
    return gradient_reg

# load data
data = np.matrix(np.loadtxt('ex2data2.txt', delimiter=','))

X = data[:, 0:2]
y = data[:, 2]

# plot data
plot_data(X, y)
plt.show()



# part 1: regularized logistical regression
# dataset is not linearly separable, so we add polynomial features to the data

# add polynomial features, including intercept term
X = map_features(X[:,0], X[:,1])

[m, n] = X.shape


# initialize fitting parameters
theta = np.zeros([1, n])

# initialize regularization parameter
λ = 1

# compute cost and gradient at initial theta
cost = cost_function(theta, X, y, λ)
grad = gradient_function(theta, X, y, λ)

print("Cost at initial theta (zeros):", cost)

# Part 2: regularization and accuracies


