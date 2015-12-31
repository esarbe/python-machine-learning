import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plot import plotData
from data import loadData
import GradientDescent as gd
import numpy as np

data = loadData('ex1data1.txt')

(m, features) = data.shape

X = data[:, 0]
y = data[:, 1]

# Plotting the data
plotData(X, y)
plt.show()

# Gradient Descent
print("Running Gradient Descent ...")

X = np.append(np.ones((m, 1)), X, 1) # add a column of ones to X
theta = np.zeros((2, 1)) # initialize fitting parameters

# some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost (expected: 32.0727338775)
cost = gd.computeCost(X, y, theta)
print("initial cost: ", cost)

# run gradient descent
(theta, history) = gd.performGradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print("theta found by gradient descent: ", theta.A1)

# plot original data (plt.hold(True) doesn't seem to work?...)
plotData(X[:, 1], y)
# plot linear fit from gradient descent
plt.plot(X[:, 1], X * theta, '-')
plt.legend(["Training data", "Linear regression"])
plt.show()


# Predict values for population sizes of 35,000 and 70,000
prediction0 = np.matrix([1, 3.5]) * theta
print("for population of 35'000 we predict a profit of", prediction0.A1 * 10000)

prediction1 = np.matrix([1, 7]) * theta
print("for population of 70'000 we predict a profit of", prediction1.A1 * 10000)

# visualizing J(theta_0, theta_1)
print("Visualizing J(theta_0, theta_1)")

# Grid over which we calculate J
theta_0_vals = np.linspace(-10, 10, 100)
theta_1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0s
J_vals = np.zeros((theta_0_vals.size, theta_1_vals.size))

for i in range(theta_0_vals.size):
  for j in range(theta_1_vals.size):
    t = [[theta_0_vals[i]], [theta_1_vals[j]]]
    J_vals[i, j] = gd.computeCost(X, y, np.matrix(t))

# because of the way the grids are rendered using plot_surface and contour we
# have to transpose the J_vals
J_vals = J_vals.T

X, Y = np.meshgrid(theta_0_vals, theta_1_vals)
plt.figure().gca(projection = '3d').plot_surface(X, Y, J_vals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()


plt.figure()
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
levels = np.logspace(-2, 3, 20)
plt.contour(X, Y, J_vals, levels)
plt.plot(theta[0], theta[1], 'rx')
plt.show()

