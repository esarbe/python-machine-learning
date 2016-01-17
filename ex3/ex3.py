import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

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

# part 1: loading and visualization
data = sio.loadmat('ex3data1.mat')


X = data['X']
m, n = X.shape

plot_data(np.random.permutation(X)[:49,:])
