from numpy import * 

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
	
	dataNormalized = (data - mu) / sigma

	return dataNormalized, mu, sigma
