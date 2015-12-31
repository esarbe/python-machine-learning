from numpy import mean, std, matrix


def parseLine(line):
  return [ float(value) for value in line ] 

def loadData(filename, parseLine = parseLine):
  data = []
  with open(filename) as lines: 
    for line in lines:
      rawData = line.split(",")
      parsedData = parseLine(rawData) 
      data.append(parsedData)
    return matrix(data)

def normalizeFeatures(data):
  mu = mean(data, 0)
  sigma = std(data, 0, ddof=1)

  data_normalized = (data - mu) / sigma

  return data_normalized, mu, sigma
