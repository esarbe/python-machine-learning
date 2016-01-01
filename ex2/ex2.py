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

def plotData(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    print("pos:", pos)
    print("v:", X[pos, 0:1])
    plt.plot(X[pos, :], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg, :], 'ko', markersize=7)


data = np.matrix(load('ex2data1.txt'))

X = data[:, 0:2]
y = data[:, 2]
print(X)

plotData(X, y)
plt.show()


