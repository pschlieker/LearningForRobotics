import pandas as pd
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

#Calculates Euclidian Distance between two points
def calculateDistance(x,y, indexOfAttributes):
	dis = 0
	#Compute the euclidian distance, exclude id field in first column and class in last column
	for i in range(indexOfAttributes[0], indexOfAttributes[1]+1):
		dis = dis + np.power(x[i] - y[i],2)
	return(np.sqrt(dis))

#Maps each value in c to a color and returns array of same length than c containing color
#for each value in c 
def getColors(c):
	colors=['royalblue','forestgreen','deepskyblue','indianred', 'limegreen', 'darkorange']

	classNames = np.unique(c)
	classesLookup = {x:i for i,x in enumerate(classNames)}

	r = [None] * c.shape[0]#c.shape[0], dtype='str')
	for i in range(c.shape[0]):
		r[i] = colors[classesLookup.get(c[i])]

	return r

def plot3D(x, y, z, c, axisNames, title):
	colors = getColors(c)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z, c=colors)
	ax.set_xlabel(axisNames[0])
	ax.set_ylabel(axisNames[1])
	ax.set_zlabel(axisNames[2])	
	ax.set_title(title)

	cT = mpatches.Patch(color='royalblue', label='True')
	cF = mpatches.Patch(color='forestgreen', label='False')
	cTP = mpatches.Patch(color='deepskyblue', label='TP')
	cFN = mpatches.Patch(color='indianred', label='FN')
	cTN = mpatches.Patch(color='limegreen', label='TN')
	cFP = mpatches.Patch(color='darkorange', label='FP')

	plt.legend(handles=[cT,cF, cTP, cFP, cTN, cFN])
	plt.show()

def plot2D(x, y, c, axisNames, title):
	colors = getColors(c)

	scatter = plt.scatter(x, y, c=colors)
	plt.xlabel(axisNames[0])
	plt.ylabel(axisNames[1])
	plt.title(title)

	cT = mpatches.Patch(color='royalblue', label='Positive')
	cF = mpatches.Patch(color='forestgreen', label='Negative')
	cTP = mpatches.Patch(color='deepskyblue', label='TP')
	cFN = mpatches.Patch(color='indianred', label='FN')
	cTN = mpatches.Patch(color='limegreen', label='TN')
	cFP = mpatches.Patch(color='darkorange', label='FP')

	plt.legend(handles=[cT,cF, cTP, cFP, cTN, cFN])
	plt.show()

def plot(data, plotValues, axisNames, title):
	if(len(plotValues) == 3):
		axis = [axisNames[plotValues[0]],axisNames[plotValues[1]]]
		plot2D(data[:,plotValues[0]], data[:,plotValues[1]], data[:,plotValues[2]], axis, title)
	if(len(plotValues) == 4):
		axis = [axisNames[plotValues[0]],axisNames[plotValues[1]],axisNames[plotValues[2]]]
		plot3D(data[:,plotValues[0]], data[:,plotValues[1]], data[:,plotValues[2]], data[:,plotValues[3]], axis, title)

#Predicts using knn and returns data containing the predictions
def knn(k, path, indexOfAttributes, indexOfClass):
	#Read Data from csv
	data = pd.read_csv(path, delimiter=",", na_values="?", header=None, index_col=False)
	data = np.array(data)

	#Create train & test set
	np.random.shuffle(data) #Shuffle data set
	trainingSize = int(float(testTrainingRatio) * data.shape[0])
	train, test = data[:trainingSize,:], data[trainingSize:,:]

	#Store shape of data for easy access
	l = test.shape[0] # rows
	m = train.shape[0] # rows
	n = train.shape[1] # columns

	#Get Classes
	classNames = np.unique(train[:,indexOfClass])

	#Create dictionary to lookup classes
	classes = {x:i for i,x in enumerate(classNames)}

	#Create Confusion Matrix
	confMatrix = np.zeros((len(classes),len(classes)))

	#Store Predictions
	predictions = np.empty(l)

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

		#Check prediction
		prediction = (cnt.most_common(1))[0][0]
		confMatrix[classes.get(prediction)][classes.get(test[toPredict][indexOfClass])]+=1

		#Store Details about the prediction as int
		#Prediction - ActualClass
		#11 => Prediction was 1, true class was 1
		#21 => Prediction was 2, true class was 1
		predictions[toPredict] = int(str(int(prediction))+str(int(test[toPredict][indexOfClass])))

	#Print Results
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

	#replace expected results with predictions 
	data[:,indexOfClass] = np.concatenate((train[:,indexOfClass], predictions))
	return data


#Settings
testTrainingRatio = 0.8 #Part used for Training 

habermanAxisNames = ["Age", "YearOperation", "AuxilliaryNodes", "Class"]
wisconsinAxisNames = ["SampleNumber", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]

data = knn(3, "haberman.data", (0,2), 3)
knn(6, "haberman.data", (0,2), 3)
plot(data, [0,1,3], habermanAxisNames, "haberman k=3")

data = knn(3, "breast-cancer-wisconsin.data", (1,9), 10)
knn(6, "breast-cancer-wisconsin.data", (1,9), 10)
plot(data, [1,6,10], wisconsinAxisNames, "Wisconsin k=3")
plot(data, [1,6,8,10], wisconsinAxisNames,"Wisconsin k=3")