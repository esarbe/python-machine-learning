import matplotlib.pyplot as plt

def plotData(xArr, yArr):
  plt.plot(xArr, yArr, 'rx')
  plt.xlabel("Population of City in 10'000s")
  plt.ylabel("Profit in $10'000s")

