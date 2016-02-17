# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:36:56 2016

@author: kanchan


THIS PROGRAM LOADS DATA FROM .MAT (MATLAB VERSION <7.3 ( DOESN'T SUPPORT 7.3) ) TO PYTHON USING SCIPY.IO and then train a neural 
network using sknn.mlp package in python 
"""

import numpy as np
import scipy.io as sio
from sknn.mlp import Classifier, Layer

x = sio.loadmat('input1_ss3_scrM100_v7.mat')
X_train_set = x['input1']
Y_train_set = x['target1']

print "X_train_set = ",X_train_set.shape
print "Y_train_set = ",Y_train_set.shape

cutoff=int((X_train_set.size/X_train_set[0].size)*0.7) 
print "cutoff at  ",cutoff


X_train = X_train_set[0:cutoff,:]
X_test = X_train_set[cutoff:,:]
print X_test.size/X_test[0].size
#X_validate= X_train_set[15000:20000,:]

Y_train = Y_train_set[0:cutoff,:]
Y_test = Y_train_set[cutoff:,:]
#Y_validate= Y_train_set[15000:20000,:]

#print X_train.shape
print "X_train shape = ",X_train.shape
print "Y_train shape = ",Y_train.shape
print "X_test shape = ", X_test.shape
print "Y_test shape = ", Y_test.shape
print "Y_test size = ", Y_test.shape[0]
#print (X_train).shape



nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid")],
    learning_rate=0.001,
    n_iter=20,verbose=True,debug = True)



print "lets make data fit"
nn.fit(X_train, Y_train)

print "fitting ends here and prediction starts"
Y_example= nn.predict(X_test)

np.savetxt("Y_example_prediction_protein2.out ",Y_example)


match= np.logical_and(Y_test,Y_example).sum()
print "Y_"
percent_match = (match*100.0)/Y_test.shape[0]
print "match = " ,match
print "percentage_match = ", percent_match,"%" # for 20 iterations the percentage match was 65.2323817404 %

