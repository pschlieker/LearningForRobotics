import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV

#Split dataset into training et testing sets
dir_path = "./dataset/"
train_set = pd.read_csv(dir_path + "mnist_train.csv")
train_set = np.array(train_set)
test_set = pd.read_csv(dir_path + "mnist_test.csv")
test_set = np.array(test_set)

#Shufle data to be able to use parts as smaller training set
np.random.shuffle(train_set)
np.random.shuffle(test_set)

#Split X & y and normalize
X_train = train_set[:,1:]/255.0
y_train = train_set[:,0]
X_test = test_set[:,1:]/255.0
y_test = test_set[:,0]

################################
####### PARAMETER SEARCH #######

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 'C': [0.1, 0.5, 1, 5, 10, 50]},
#                     {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel':['poly'], 'degree':[0, 1, 2, 3, 4, 5, 6] },
#                     {'kernel': ['linear'], 'C': [0.00001, 0.0001, 0.001, 0.10, 1.0, 2.0 ]}]

# clfSVM = svm.SVC()
# clf = GridSearchCV(clfSVM, tuned_parameters, cv=5, n_jobs=30)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Best parameters: "+str(clf.best_params_))

# Result:
# Best parameters: {'C': 5, 'gamma': 0.05, 'kernel': 'rbf'}


###############################
######## OPTIMIZED SVM ########

clf = svm.SVC(C=5,gamma=0.05, kernel='rbf')
clf.fit(X_train, y_train)

#Prediction
y_pred = clf.predict(X_test)

#Results
print("Accuracy score: "+ str(accuracy_score(y_test, y_pred) * 100))
print("\nConfusion matrix: \n" + str(confusion_matrix(y_test, y_pred)))

#####################################
######## EXPECTED RESULT SVM ########

# Accuracy score: 98.37

# Confusion matrix:
# [[ 973    0    1    0    0    2    1    1    2    0]
#  [   0 1127    3    1    0    1    0    1    2    0]
#  [   4    0 1015    0    1    0    0    6    6    0]
#  [   0    0    2  995    0    3    0    6    4    0]
#  [   0    0    3    0  966    0    4    0    2    7]
#  [   2    0    0    5    1  878    2    1    2    1]
#  [   4    2    0    0    2    3  946    0    1    0]
#  [   0    3   10    1    1    0    0 1004    2    7]
#  [   1    0    1    4    1    2    0    2  960    3]
#  [   3    3    2    6    9    2    0    5    6  973]]
