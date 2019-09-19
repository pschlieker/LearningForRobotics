import pandas as pd
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculateDistance(x,y, indexOfAttributes):
	dis = 0
	#Compute the euclidian distance, exclude id field in first column and class in last column
	for i in range(indexOfAttributes[0], indexOfAttributes[1]+1):
		dis = dis + np.power(x[i] - y[i],2)
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

def knn(k, path, indexOfAttributes, indexOfClass):
	#Read Data from csv
	data = pd.read_csv(path, delimiter=",", na_values="?", header=None, index_col=False)
	data = np.array(data)

	#Create train & test set
	np.random.shuffle(data) #Shuffle data set
	trainingSize = int(float(testTrainingRatio) * data.shape[0])
	train, test = data[:trainingSize,:], data[trainingSize:,:]

	#Store shape of data for easy access
	l = test.shape[0] # columns
	m = train.shape[0] # rows
	n = train.shape[1] # columns

	#Get Classes
	classNames = np.unique(train[:,indexOfClass])

	#Create dictionary to lookup classes
	classes = {x:i for i,x in enumerate(classNames)}

	#Create Confusion Matrix
	confMatrix = np.zeros((len(classes),len(classes)))

	#Select element to predict
	for toPredict in range(l):
		#Create array to store distances to that point
		dist = np.empty(m)

		#Calculate distances to all points
		for x in range(0, m):
			dist[x] = calculateDistance(train[x], test[toPredict], indexOfAttributes);

		#Sort Points by distance
		closestByIndex = dist.argsort()

		#Calculate Class from k-nn
		cnt = Counter((train[closestByIndex])[:k,indexOfClass])

		#Store predictions marking if they are correct or wrong
		predictions = (cnt.most_common(1))[0][0]

		confMatrix[classes.get(predictions)][classes.get(test[toPredict][indexOfClass])]+=1

		#Mark wrong predictions
		if test[toPredict][indexOfClass] == predictions:
			test[toPredict][indexOfClass] = -1

	print("######## RESULTS k = "+str(k)+" FOR "+path+" ########")

	print("Accuracy: "+str((confMatrix[0][0] + confMatrix[1][1]) / l * 100)+"%")
	print()
	print("        CONFUSSION MATRIX")
	print("          ACTUAL CLASS       ")
	print("\t\t\t"+str(classNames[0])+"\t\t"+str(classNames[1]))
	print("PREDI  "+str(classNames[0])+"\t" +str(confMatrix[0][0])+"\t"+str(confMatrix[0][1]))
	print("CTION  "+str(classNames[1])+"\t"+str(confMatrix[1][0])+"\t"+str(confMatrix[1][1]))
	print()
	print()

#Settings
testTrainingRatio = 0.2 #Part used for testing 

knn(3, "haberman.data", (0,2), 3)
knn(6, "haberman.data", (0,2), 3)

knn(3, "breast-cancer-wisconsin.data", (1,9), 10)
knn(6, "breast-cancer-wisconsin.data", (1,9), 10)


#plot2D(1,2,indexOfClass)
#plot2D(0,2,indexOfClass)
#plot2D(0,1,indexOfClass)

#plot3D(0,1,2,indexOfClass)