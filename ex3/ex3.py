import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize
import math


def logistic_regression_cost_function(theta, X, y, λ):
    theta = np.matrix(theta)
    m, n = X.shape
    cost = (1 / m ) * (-y.T * np.log(sigmoid(X * theta.T)) - ( 1 - y ).T * np.log( 1 - sigmoid(X * theta.T)))
    cost = cost.A1
    reg_cost = cost + (λ / 2 / m) * np.sum(np.power(theta[:,1:], 2))
    return reg_cost[0]


def logistic_regression_gradient_function(theta, X, y, λ):
    theta = np.matrix(theta)
    [m, n] = X.shape
    gradient = ((sigmoid(X * theta.T) - y).T * X).T / m;
    theta[0,0] = 0
    reg = (λ / m) * theta
    gradient_reg = gradient + reg.T
    return gradient_reg.A1


def one_vs_all(X, y, num_labels, λ):
    m, n = X.shape
    X = np.hstack([np.ones([m, 1]), X])
    thetas = []
    initial_theta = np.matrix(np.zeros([n + 1, 1]))
    for label in range(1, num_labels + 1):
        theta = optimize.fmin_cg(
            logistic_regression_cost_function,
            initial_theta,
            logistic_regression_gradient_function,
            args=(X, (y == label).astype(float), λ))
        thetas.append(theta)

    return thetas

def sigmoid(z):
    return 1 / ( 1 + np.power(np.e, -z))

def predict_all_vs_one(all_theta, X):
    m, n = X.shape
    X = np.hstack([np.ones([m, 1]), X])
    xt = X * np.matrix(all_theta).T

    return xt.argmax(axis=1)


def plot_data(X, item_width=None):
    [m, n] =  X.shape
    if not item_width:
        item_width = np.round(np.sqrt(n))

    item_height = (n / item_width)

    pad = 1

    # grid of items to display
    rows = math.floor(np.sqrt(m))
    cols = math.ceil(m / rows)

    display_array = - np.ones([pad + rows * (item_height + pad),
                              pad + rows * (item_width  + pad)])

    current_item = 0
    for row in range(rows):
        for col in range(cols):
            offset_x = pad + row * (item_width + pad)
            offset_y = pad + col * (item_height + pad)

            data = X[current_item,:].reshape([item_height, item_width])
            t = display_array[offset_x:offset_x + item_width, offset_y: offset_y + item_height]
            display_array[offset_x:offset_x + item_width, offset_y: offset_y + item_height] = data

            current_item = current_item + 1

    plt.imshow(display_array.T, cmap = cm.Greys_r)
    plt.show()

input_layer_size = 400
num_labels = 10
λ = 0.1

# part 1: loading and visualization
data = sio.loadmat('ex3data1.mat')

X = data['X']
y = data['y']
m, n = X.shape

plot_data(np.random.permutation(X)[:100,:])

# part2:

all_thetas = one_vs_all(X, y, num_labels, λ)

print("all_thetas", all_thetas)

# part 3:
pred = predict_all_vs_one(all_thetas, X)

print("Training set accuracy", np.mean((pred == y).astype(float)) * 100)
