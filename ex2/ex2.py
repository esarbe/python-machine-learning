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

def plot_decision_boundary(theta, X, y):
    plot_data(X, y)
    theta = np.array(theta)
    plot_x = np.array([np.min(X[:,0]), np.max(X[:,0])])
    plot_y = (-1 / theta[2]) * ((theta[1] * plot_x) + theta[0])
    plt.plot(plot_x, plot_y)

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

def predict(theta, X):
    pred = [ 1 if x > 0.5 else 0 for x in sigmoid(np.matrix(theta) * X.T).flat]
    return pred

data = np.matrix(np.loadtxt('ex2data1.txt', delimiter=','))

X = data[:, 0:2]
y = data[:, 2]


# part 1: plotting data
plot_data(X, y)
plt.show()

# part 2: compute gradient and cost
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

# part 3: optimize using fmin

theta = optimize.fmin(cost_function, x0 = initial_theta, args = (X, y))
cost = cost_function(theta, X, y)
print('cost at theta found by fmin:\n\t', cost)

# part 4: predict and accuracies
plot_decision_boundary(theta, X[:, 1:3], y)
plt.show()

prob = sigmoid(np.matrix([1., 45., 85.]) * np.matrix(theta).T)[0,0]
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)

prediction = predict(theta, X)
accuracy = np.mean((np.array(y).T == prediction).astype(float))
print("Train accuracy:\n\t", accuracy * 100)
