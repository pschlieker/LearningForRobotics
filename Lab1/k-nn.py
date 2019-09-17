import pandas as pd
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Settings
k = 5
runs = 1000

#Haberman
#path = "haberman.data"
#indexOfAttributes = (0,2)
#indexOfClass = 3
#axisNames = ["Age", "YearOperation", "AuxilliaryNodes", "Class"]

#Breast Cancer Wisconsin
path = "breast-cancer-wisconsin.data"
indexOfAttributes = (1,9)
indexOfClass = 10
axisNames = ["SampleNumber", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]

def calculateDistance(x,y):
	dis = 0
	#Compute the euclidian distance, exclude id field in first column and class in last column
	for i in range(indexOfAttributes[0], indexOfAttributes[1]+1):
		dis = dis + (x[i] - y[i])**2
	return(np.sqrt(dis))

def plot3D(xaxisIndex, yaxisIndex, zaxisIndex, classIndex):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:,xaxisIndex], data[:,yaxisIndex], data[:,zaxisIndex], c=data[:,classIndex])
	ax.set_xlabel(axisNames[xaxisIndex])
	ax.set_ylabel(axisNames[yaxisIndex])
	ax.set_zlabel(axisNames[zaxisIndex])
	plt.show()

def plot2D(xaxisIndex, yaxisIndex, classIndex):
	plt.scatter(data[:,xaxisIndex], data[:,yaxisIndex], c=data[:,classIndex])
	plt.xlabel(axisNames[xaxisIndex])
	plt.ylabel(axisNames[yaxisIndex])
	plt.show()

#Read Data from csv
data = pd.read_csv(path, delimiter=",", na_values="?", header=None, index_col=False)
data = np.array(data)

#Store shape of data for easy access
n = data.shape[1] # columns
m = data.shape[0] # rows

#Get Classes
classesVal = np.unique(data[:,indexOfClass])

#Create dictionary to lookup classes
classes = {x:i for i,x in enumerate(classesVal)}

#Create Confusion Matrix
confMatrix = np.zeros((len(classes),len(classes)))

for r in range(runs):
	#Select element to predict
	toPredict = random.randint(0,m-1)

	#Create array to store distances to that point
	dist = np.empty(m)

	#Calculate distances to all points
	for x in range(0, m):
		dist[x] = calculateDistance(data[x], data[toPredict]);

	#Set distance to point to predict to infinity
	dist[toPredict] = np.Infinity

	#Sort Points by distance
	closestByIndex = dist.argsort()

	#Calculate Class from k-nn
	cnt = Counter((data[closestByIndex[::-1]])[:k,indexOfClass])
	prediction = (cnt.most_common(1))[0][0]

	confMatrix[classes.get(prediction)][classes.get(data[toPredict][indexOfClass])]+=1



print("Accuracy: "+str((confMatrix[0][0] + confMatrix[1][1]) / runs * 100)+"%")

print("###  CONFUSSION MATRIX ###")
print("          ACTUAL CLASS       ")
print("\t\t\t"+str(classesVal[0])+"\t\t"+str(classesVal[1]))
print("PREDI  "+str(classesVal[0])+"\t" +str(confMatrix[0][0])+"\t"+str(confMatrix[0][1]))
print("CTION  "+str(classesVal[1])+"\t"+str(confMatrix[1][0])+"\t"+str(confMatrix[1][1]))

plot2D(1,2,indexOfClass)
plot3D(0,1,2,indexOfClass)