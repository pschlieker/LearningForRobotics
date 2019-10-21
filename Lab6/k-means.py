import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from PIL import Image
import random
import matplotlib.pyplot as plt

#Split dataset into training et testing sets
dir_path = "./dataset/"
train_set = pd.read_csv(dir_path + "optdigits.tra")
train_set = np.array(train_set)
test_set = pd.read_csv(dir_path + "optdigits.tes")
test_set = np.array(test_set)

#Shufle data to be able to use parts as smaller training set
np.random.shuffle(train_set)
np.random.shuffle(test_set)

#Split X & y
X_train = train_set[:,:64]
y_train = train_set[:,64]
X_test = test_set[:,:64]
y_test = test_set[:,64]

#Init centroids
centroids = np.zeros(shape=(10,64))

for i in range(10):
    centroids[i] = random.choice(X_train[y_train == i])

#Perform K-Means
kmeans = KMeans(n_clusters = 10, init = centroids)
y_train_pred = kmeans.fit_predict(X_train)
y_test_pred = kmeans.predict(X_test)

#Output Accuracy
print("Accuracy score on training set: "+ str(accuracy_score(y_train, y_train_pred) * 100))
print("Accuracy score on testing set: "+ str(accuracy_score(y_test, y_test_pred) * 100))
print("Confusion matrix on testing set: \n" + str(confusion_matrix(y_test, y_test_pred)))

#Display prototype of each digit
for i in range(10):
    prot = np.mean(X_test[y_test == i], axis = 0)
    prot = np.reshape(prot, (8,8))

    if i == 0:
        prototype = prot
    else:
        prototype = np.concatenate((prototype, prot), axis=1)

plt.matshow(prototype, cmap='gray')
plt.show()