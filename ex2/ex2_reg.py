import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def plot_data(X, y, label=["x=1", "x=0"]):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos, 0].flat, X[pos, 1].flat, 'k+', markersize=7)
    plt.plot(X[neg, 0].flat, X[neg, 1].flat, 'yo', markersize=7)
    plt.legend(label, numpoints=1)
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")

def plot_decision_boundary(theta):
    x0s = np.linspace(-1, 1.5, 6)
    x1s = np.linspace(-1, 1.5, 6)
    Z = np.zeros([len(x0s), len(x1s)])
    # evaluate z = theta*x over the grid
    for x0 in x0s:
        for x1 in x1s:
            #import pdb; pdb.set_trace()
            Z[x0, x1] = map_features(np.matrix(x0), np.matrix(x1)) * theta.T
            print(x0, "/", x1, ":", Z[x0, x1])
    X, Y = np.meshgrid(x0s, x1s)
    print("x0s", x0s)
    print("Z", Z)
    print("Z min", Z.min(), ", Z max:", Z.max())
    plt.contour(X, Y, Z.T)

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
    return reg_cost[0]


def gradient_function(theta, X, y, λ):
    theta = np.matrix(theta)
    [m, n] = X.shape
    gradient = ((sigmoid(X * theta.T) - y).T * X).T / m;
    theta[0,0] = 0
    reg = (λ / m) * theta
    gradient_reg = gradient + reg.T
    return gradient_reg.A1


def predict(theta, X):
    pred = [ 1 if x > 0.5 else 0 for x in sigmoid(np.matrix(theta) * X.T).flat]
    return pred

# load data
data = np.matrix(np.loadtxt('ex2data2.txt', delimiter=','))

X = data[:, 0:2]
y = data[:, 2]

print("y min:", y.min(), " y max:", y.max())

# plot data
plot_data(X, y)
plt.show()

# part 1: regularized logistical regression
# dataset is not linearly separable, so we add polynomial features to the data

# add polynomial features, including intercept term
X_mapped = map_features(X[:,0], X[:,1])

[m, n] = X_mapped.shape

# initialize fitting parameters
theta = np.matrix(np.zeros([1, n]))

# initialize regularization parameter
λ = 1

# compute cost and gradient at initial theta
cost = cost_function(theta, X_mapped, y, λ)
grad = gradient_function(theta, X_mapped, y, λ)

print("Cost at initial theta (zeros):", cost)

# Part 2: regularization and accuracies
initial_theta = theta

theta = optimize.fmin_bfgs(cost_function, initial_theta, gradient_function, args = (X_mapped, y, λ))
cost = cost_function(theta, X_mapped, y, λ)

print('cost at theta found by fmin:\n\t', cost)

plot_decision_boundary(np.matrix(theta))
#plot_data(X,y, label=["Decision boundary", "X=1", "X=0"])
plt.show()

prediction = predict(theta, X_mapped)

accuracy = np.mean((np.array(y).T == prediction).astype(float))
print("Training set accuracy:\n\t", accuracy * 100)
