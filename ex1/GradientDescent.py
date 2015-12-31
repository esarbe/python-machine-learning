from numpy import sum, power


def performGradientDescent(X, y, theta, alpha, num_iterations):
  m = y.size
  J_history = []
  for i in range(num_iterations):
    theta = theta - ((alpha / m) * ((X * theta) - y).T * X).T
    J = computeCost(X, y, theta)
    J_history.append(J)

  return (theta, J_history)

def computeCost(X, y, theta):
  m = y.size
  return sum(power(X * theta - y, 2)) / (2 * m)
