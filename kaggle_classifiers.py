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

train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv',header=0)
train2 = train.copy()  #keep a copy of training data

x_train = train.drop(['id', 'species'], axis=1)
y_train = train.pop('species')
x_test = test.drop(['id'],axis=1)
test_ids = test.pop('id')

#labelencoder may not be useful in this case as the labels are not ordinal

scaler = StandardScaler().fit(x_train) #find mean and std for the standardization
x_train = scaler.transform(x_train)  #standardize the training values
x_test = scaler.transform(x_test)

#Initialise the K-fold with k=5
numFold = 5
kfold = KFold(n_splits=numFold, shuffle=True, random_state=4)

print("====================================================================================================================")
#Initialise Naive Bayes
classNB = GaussianNB()
#We can now run the K-fold validation on the dataset with Naive Bayes
#this will output an array of scores, so we can check the mean and standard deviation

#nb_validation=[nb.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean() \
        #   for train, test in kfold.split(x_train)]
score_NB = []

for train, test in kfold.split(x_train):
    classNB.fit(x_train[train], y_train[train])
    score = classNB.score(x_train[test],y_train[test])
    score_NB.append(score)

mean_NB = sum(score_NB)/numFold
print("The average accuracy of Naive Bayes is : {:.4%}".format(mean_NB))

print("=========================================================================================================================")
'''
classET = ExtraTreesClassifier(n_estimators=500, random_state=0)
score_ET = []

for train, test in kfold.split(x_train):
    classET.fit(x_train[train], y_train[train])
    score = classET.score(x_train[test],y_train[test])
    score_ET.append(score)

mean_ET = sum(score_ET)/numFold
print("The average accuracy of Extra Trees is : {:.4%}".format(mean_ET))

print("==========================================================================================================================")
'''
classDT = DecisionTreeClassifier()
score_DT = []

for train, test in kfold.split(x_train):
    classDT.fit(x_train[train], y_train[train])
    score = classDT.score(x_train[test],y_train[test])
    score_DT.append(score)

mean_DT = sum(score_DT)/numFold
print("The average accuracy of Decision Tree is : {:.4%}".format(mean_DT))

print("==========================================================================================================================")

classKNN = KNeighborsClassifier(10)
score_KNN = []

for train, test in kfold.split(x_train):
    classKNN.fit(x_train[train], y_train[train])
    score = classKNN.score(x_train[test],y_train[test])
    score_KNN.append(score)

mean_KNN = sum(score_KNN)/numFold
print("The average accuracy of KNN is : {:.4%}".format(mean_KNN))

print("==========================================================================================================================")
'''
classANN = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(50,192),batch_size=16,max_iter=50)
score_ANN = []

for train, test in kfold.split(x_train):
    classANN.fit(x_train[train], y_train[train])
    score = classANN.score(x_train[test],y_train[test])
    score_ANN.append(score)

mean_ANN = sum(score_ANN)/numFold
print("The average accuracy of ANN is : {:.4%}".format(mean_ANN))

print("==========================================================================================================================")
'''
# Prediction
final_classifier = classKNN
#for train, test in kfold2.split(x_train):
#    final_classifier.fit(x_train[train],y_train[train])

#ensemble
'''
bag = BaggingClassifier(n_estimators =50 ,base_estimator=KNeighborsClassifier(),max_samples = 0.5,max_features = 0.5)
bag.fit(x_train, y_train)
#prediction = bag.predict(x_test)
prob = bag.predict_proba(x_test)
'''
#ada = AdaBoostClassifier(n_estimators = 100,base_estimator=GaussianNB(),learning_rate = 1.0)
#ada.fit(x_train, y_train)
#prob = ada.predict_proba(x_test)

#final_classifier.fit(x_train,y_train)
#prob = final_classifier.predict_proba(x_test)

classANN = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(50,192),batch_size=16,max_iter=120)
#classANN.fit(x_train,y_train)
#prob = classANN.predict_proba(x_test)

bag = BaggingClassifier(n_estimators =30,base_estimator=classANN,max_samples = 0.5,max_features = 0.5)
bag.fit(x_train,y_train)
prob = bag.predict_proba(x_test)
# Format DataFrame
submission = pd.DataFrame(prob, columns=sorted(train2.species.unique()))
#submission = pd.DataFrame(prob, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('kaggle25submission.csv', index = False)
submission.tail()
