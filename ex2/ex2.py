import numpy as np
import matplotlib.pyplot as plt


def load(filename):
  import csv

  data = []
  with open(filename, 'r') as source:
    reader = csv.reader(source, delimiter=',')
    for row in reader:
      data.append([ float(x) for x in row])
  return data

def plot_data(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    plt.plot(X[pos][0][:,0], X[pos][0][:,1], 'k+', label="Admitted", linewidth=2, markersize=7)
    plt.plot(X[neg][0][:,0], X[neg][0][:,1], 'yo', label="Not admitted", markersize=7)
    plt.legend(numpoints=1)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")

def sigmoid(z):
    return 1 / ( 1 + np.power(np.e, -z))

def cost_function(theta, X, y):
    [m, n] = X.shape
    cost = (1 / m ) * (-y.T * np.log(sigmoid(X * theta)) - ( 1 - y ).T * np.log( 1 - sigmoid(X * theta)))

    return cost

data = np.matrix(load('ex2data1.txt'))

X = data[:, 0:2]
y = data[:, 2]

plot_data(X, y)
plt.show()

[m, n] = X.shape

# setup data matrix; add ones for the intercept terms
X = np.hstack([np.ones([m, 1]), X])

initial_theta = np.zeros([n + 1, 1])

cost = cost_function(initial_theta, X, y)

print('initial cost:', cost)

