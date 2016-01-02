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
    plt.plot(X[pos][:,0], X[pos][:,1], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg][:,0], X[neg][:,1], 'yo', markersize=7)
    plt.legend(["Admitted", "Not admitted"], numpoints=1)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")

data = np.array(load('ex2data1.txt'))

X = data[:, 0:2]
y = data[:, 2]

plotData(X, y)
plt.show()


