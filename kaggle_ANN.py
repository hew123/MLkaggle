import csv as csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


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

'''
clfANN = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(50,192),batch_size=16,max_iter=200)
clfANN.fit(x_train,y_train)
prob = clfANN.predict_proba(x_test)
'''

clfANN = MLPClassifier(solver='sgd',activation='identity', hidden_layer_sizes=(600,192),batch_size=16,max_iter=200)
clfANN.fit(x_train,y_train)
prob = clfANN.predict_proba(x_test)

'''
#bagging did not improve score
bag = BaggingClassifier(n_estimators =10,base_estimator=clfANN,max_samples = 0.5,max_features = 0.5)
bag.fit(x_train,y_train)
prob = bag.predict_proba(x_test)
'''

'''
#voting did not improve score
clf1 = DecisionTreeClassifier(max_depth=10)
clf2 = KNeighborsClassifier(n_neighbors=10)
clf3 = clfANN
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('ann', clf3)], voting='soft', weights=[1, 2, 4])
eclf.fit(x_train,y_train)
prob = eclf.predict_proba(x_test)
'''

# Format DataFrame
submission = pd.DataFrame(prob, columns=sorted(train2.species.unique()))
#submission = pd.DataFrame(prob, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('kaggle25submission.csv', index = False)
submission.tail()
