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

data = np.matrix(np.loadtxt('ex2data2.txt', delimiter=','))

X = data[:, 0:2]
y = data[:, 2]

plot_data(X, y)
plt.show()
