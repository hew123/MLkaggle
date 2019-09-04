import csv as csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit

train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv',header=0)
train2 = train.copy()  #keep a copy of training data

x_train = train.drop(['id', 'species'], axis=1)
y_train = train.pop('species')
#x_test = test.drop(['id'],axis=1)
#test_ids = test.pop('id')

#labelencoder may not be useful in this case as the labels are not ordinal

scaler = StandardScaler().fit(x_train) #find mean and std for the standardization
x_train = scaler.transform(x_train)  #standardize the training values
#x_test = scaler.transform(x_test)

sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, test_index in sss.split(x_train, y_train):
    X_train, X_test = x_train[train_index], x_train[test_index]
    Y_train, Y_test = y_train[train_index], y_train[test_index]

clf = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(10,50),batch_size=16,max_iter=50)

print("------ normal split training data ------")

model = clf
#model = KNeighborsClassifier(10)
model.fit(X_train,Y_train)
score1 = model.score(X_test, Y_test)
print(score1)

print("=================================================================================================================")
print("----- trying k fold -----")
#Initialise the K-fold with k=5
numFold = 5
kfold = KFold(n_splits=numFold, shuffle=True, random_state=4)

classNB = clf
#classNB = KNeighborsClassifier(10)

score_NB = []

for train, test in kfold.split(x_train):
    classNB.fit(x_train[train], y_train[train])
    score = classNB.score(x_train[test],y_train[test])
    score_NB.append(score)

#mean_NB = sum(score_NB)/numFold
#print("The average accuracy of Naive Bayes is : {:.4%}".format(mean_NB))
print(score_NB)

("=================================================================================================================")
print('---- trying boosting ----')
'''
#clf = KNeighborsClassifier(10)
#clf = GaussianNB()
ada = AdaBoostClassifier(n_estimators=5,base_estimator=clf,learning_rate = 1.0)
ada.fit(X_train, Y_train)
print(ada.score(X_test,Y_test))
'''
print("seems like adaboost does not work on Gaussian")
("=================================================================================================================")
print("-----trying bagging------")

#clf2 = GaussianNB()
bag = BaggingClassifier(n_estimators =30,base_estimator=clf,max_samples = 0.5,max_features = 0.5)
bag.fit(X_train,Y_train)
print(bag.score(X_test,Y_test))
