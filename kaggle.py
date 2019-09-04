# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

#import numpy as np # linear algebra
# pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

def encode(train, test):
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)
    classes = list(label_encoder.classes_)

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop('id', axis=1)

    return train, labels, test, classes

train, labels, test, classes = encode(train, test)

# standardize train features
scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)

# split train data into train and validation
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, test_index in sss.split(scaled_train, labels):
    X_train, X_test = scaled_train[train_index], scaled_train[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


nb_features = 64 # number of features per features type (shape, texture, margin)
nb_class = len(classes)

# reshape train data
#X_train_r = np.zeros((len(X_train), nb_features, 3))
#X_train_r[:, :, 0] = X_train[:, :nb_features]
#X_train_r[:, :, 1] = X_train[:, nb_features:128]
#X_train_r[:, :, 2] = X_train[:, 128:]

# reshape validation data
#X_valid_r = np.zeros((len(X_valid), nb_features, 3))
#X_valid_r[:, :, 0] = X_valid[:, :nb_features]
#X_valid_r[:, :, 1] = X_valid[:, nb_features:128]
#X_valid_r[:, :, 2] = X_valid[:, 128:]

# Keras model with one Convolution1D layer
# unfortunately more number of covnolutional layers, filters and filters lenght
# don't give better accuracy
model = Sequential()
#model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(nb_features, 3)))
#model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(192)))
model.add(Activation('relu'))
#model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(nb_class))
model.add(Activation('softmax'))


y_train = np_utils.to_categorical(y_train, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

nb_epoch = 15
model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_data=(X_test, y_test), batch_size=16)
