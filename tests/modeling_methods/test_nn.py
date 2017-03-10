# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 03:31:57 2017

@author: jb2428

After setting backend to theano...

"""


##############################################################################
# 
##############################################################################

import numpy as np
from analytic_toolkit import simulate_two_class
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, Activation, Dropout, Flatten, Merge, LSTM, Reshape, merge, Masking
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.optimizers import RMSprop, SGD
    
##############################################################################
# TRUST TO_CATEGORICAL???
##############################################################################
    
from keras.utils.np_utils import to_categorical
print(to_categorical(np.array([-1,1,-1,1,2]),3))
from sklearn.preprocessing import LabelBinarizer
print(LabelBinarizer().fit_transform([-1,1,-1,1,2])) 
 

##############################################################################
# BASIC COMPILE/FIT TEST WITH MULTIPLE OUTPUTS
##############################################################################
  
# generate dummy data
X_train = np.random.random((200, 784))
y_train = np.random.randint(5, size=(200, 1))
y_train_ = LabelBinarizer().fit_transform(y_train)

input_dim = X_train.shape[1]
output_dim = y_train_.shape[1]
model = Sequential()
model.add(Dense(output_dim=64, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dense(output_dim=output_dim))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train_, nb_epoch=3, batch_size=16)

##############################################################################
# TEST ON BINARY CLASS DATA
##############################################################################

X_train, X_test, y_train, y_test = simulate_two_class([200,100])

lb = LabelBinarizer()

y_train_ = lb.fit_transform(y_train)
y_test_ = lb.transform(y_test)           # SHOULD TEST DATA BE FORCED**!! TO SAME AS TRAIN??!?!?!?


input_dim = X_train.shape[1]
output_dim = y_train_.shape[1]
model = Sequential()
model.add(Dense(output_dim=64, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dense(output_dim=output_dim))
model.add(Activation("softmax"))
model.compile(loss='binary_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])
model.fit(X_train.values, y_train_, nb_epoch=10, batch_size=16)
 
score = model.evaluate(X_test.values, y_test_, batch_size=16)
 
 
##############################################################################
# TEST DROPOUT
##############################################################################

 
"""
 
 
in1 = Input(shape=(X_train.shape[1],), dtype='float32', name='in1')
h1 = Dense(64, activation='relu', init='he_normal', W_regularizer=l2(0.01))(in1)
d1 = Dropout(0.5)(h1)
	
h2 = Dense(64, activation='relu', init='he_normal', W_regularizer=l2(0.01))(d1)
d2 = Dropout(0.5)(h2)
		
out = Dense(y_train.shape[1], activation='softmax', name = 'outcome', init='he_normal')(d2)
		
model = Model(input=[in1], output=out)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['fbeta_score'])  # change if > 2 classes
model.fit({'in1': X_train}, {'outcome': y_train}, nb_epoch=10, batch_size=20, shuffle=True)
"""