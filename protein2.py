# -*- coding: utf-8 -*-
"""


@author: kanchan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:00:34 2015

@author: kanchan

This program uses 5 layers 
"""

from sknn.mlp import Classifier, Layer
import numpy as np



basePath="C:/Study Stuff/fall 2016/Senior Seminar/"
inputFile= basePath + "input1_ss3_scrM100.dat" #"20 inputs.dat"
outputFile= basePath + "target_ss3_scrM100.dat"#"20 outputs.dat"

print "start np.loadtxt from input and output file"
X_train_set = np.loadtxt(inputFile,dtype="float32")
Y_train_set=np.loadtxt(outputFile, dtype="float32")

cutoff=int((X_train_set.size/X_train_set[0].size)*0.7) # 16919
print cutoff


X_train = X_train_set[0:cutoff,:]
X_test = X_train_set[cutoff:,:]
#X_validate= X_train_set[15000:20000,:]

Y_train = Y_train_set[0:cutoff,:]
Y_test = Y_train_set[cutoff:,:]
#Y_validate= Y_train_set[15000:20000,:]

#print X_train.shape
#print y_train.shape
#print (X_train).shape



nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid")],
    learning_rate=0.001,
    n_iter=100,verbose=True)

print "lets make data fit"
nn.fit(X_train, Y_train)

print "fitting ends here and prediction starts"
Y_example= nn.predict(X_test)

np.savetxt("Y_example_prediction_protein2.out ",Y_example)


match= np.logical_and(Y_test,Y_example).sum() # match is 5112
print match

print "success"
