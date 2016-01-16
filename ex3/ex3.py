import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_data(X):
    padding = 1
    plt.imshow(X[0,:].reshape([20, 20]), cmap = cm.Greys_r)
    plt.show()

input_layer_size = 400
num_labels = 10

# part 1: loading and visualization
data = sio.loadmat('ex3data1.mat')

X = data['X']
m, n = X.shape

plot_data(X)
