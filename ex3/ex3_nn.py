import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def sigmoid(z):
    return 1 / ( 1 + np.power(np.e, -z))

def predict(Theta1, Theta2, X):
    m, n = X.shape

    # first layer
    X0i = np.hstack([np.ones([m, 1]), X])
    X0o = sigmoid(X0i * Theta1.T)

    # second layer
    X1i = np.hstack([np.ones([m, 1]), X0o])
    X1o = sigmoid(X1i * Theta2.T)

    # output layer
    return np.argmax(X1o, axis=1) + 1 # account for python's 0-based index

# part 1: loading and visualization
data = sio.loadmat('../data/ex3data1.mat')

X = np.matrix(data['X'])
y = np.matrix(data['y'])
m, n = X.shape

plot_data(np.random.permutation(X)[:100,:])

# part 2: loading NN parameters and implement predict
weights = sio.loadmat('../data/ex3weights.mat')

Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

pred = predict(Theta1, Theta2, X)

print("Training set accuracy", np.mean((pred == y).astype(float)) * 100)

pool = np.random.permutation(X)

for example in pool:
    pred = predict(Theta1, Theta2, np.matrix(example))
    print('Neural Network prediction', np.mod(pred, 10))

    plot_data(np.matrix(example))


