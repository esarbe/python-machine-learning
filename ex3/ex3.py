import numpy as np
import scipy.io as sio

input_layer_size = 400
num_labels = 10


# part 1: loading and visualization
data = sio.loadmat('ex3data1.mat')

m, n = data['X'].shape

print(m)
