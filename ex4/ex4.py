import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def sigmoid(z):
    return 1 / ( 1 + np.power(np.e, -z))


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

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, λ):
    m, n = X.shape
    Theta1_size = (input_layer_size + 1) * hidden_layer_size
    Theta1_shape = [hidden_layer_size, input_layer_size + 1]
    Theta2_shape = [num_labels, hidden_layer_size + 1]

    y_matrix = np.eye(num_labels)[np.mod(y, 10)][:,0,:]

    Theta1 = np.matrix(nn_params[:Theta1_size].reshape(Theta1_shape))
    Theta2 = np.matrix(nn_params[Theta1_size:].reshape(Theta2_shape))

    pred = predict(Theta1, Theta2, X)

    import pdb; pdb.set_trace()
    J_0 = np.multiply(0 - y_matrix, np.log(pred))
    J_1 = np.multiply(1 - y_matrix, np.log(1 - pred))

    return np.sum(J_0 - J_1) / m


def predict(Theta1, Theta2, X):
    m, n = X.shape

    print("Theta1.shape:", Theta1.shape, ", Theta2.shape:", Theta2.shape)
    # first layer
    a1i = np.hstack([np.ones([m, 1]), X])
    a2 = sigmoid(a1i * Theta1.T)

    # second layer
    a2i = np.hstack([np.ones([m, 1]), a2])
    a3 = sigmoid(a2i * Theta2.T)
    return a3

num_labels = 10
input_layer_size = 400
hidden_layer_size = 25
λ = 1.0
# part 1: loading and visualization
print("Loading and visualizing data")
data = sio.loadmat('../data/ex3data1.mat')

X = np.matrix(data['X'])
y = np.matrix(data['y'])
m, n = X.shape

plot_data(np.random.permutation(X)[:100,:])

# part 2: loading NN parameters
weights = sio.loadmat('../data/ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']


# unroll parameters
nn_params = np.hstack([Theta1.flat, Theta2.flat])

print("Theta1.shape:", Theta1.shape, ", Theta2.shape:", Theta2.shape)
# part 3: compute cost (feed forward)
J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, λ)

print("Cost as parameters (loaded from ex3weights.mat):\n\t", J, " (should be about 0.287629")
