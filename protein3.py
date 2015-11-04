# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03 17:06:48 2015

@author: kanchan
"""


from sknn.mlp import Classifier, Layer
import numpy as np



basePath="C:/Study Stuff/fall 2016/Senior Seminar/"
inputFile= basePath + "input1_ss3_scrM100.dat" #"20 inputs.dat"
outputFile= basePath + "target_ss3_scrM100.dat"#"20 outputs.dat"

print "start np.loadtxt from input and output file"
X_train_set = np.loadtxt(inputFile,dtype="float32")
Y_train_set=np.loadtxt(outputFile, dtype="float32")

cutoff=int((X_train_set.size/X_train_set[0].size)*0.7) 
print cutoff


X_train = X_train_set[0:cutoff,:]
X_test = X_train_set[cutoff:,:]
test_size = X_test.size/X_test[0].size


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
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid", units=100),
        Layer("Sigmoid")],
    learning_rate=0.001,
    n_iter=100,verbose=True)


print "lets make data fit"
nn.fit(X_train, Y_train)

import sys
import logging

logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)
            
import pickle
pickle.dump(nn,open('nn.pkl','wb'))

print "fitting ends here and prediction starts"
Y_example= nn.predict(X_test)

np.savetxt("Y_example_prediction_protein3.out ",Y_example)


match= np.logical_and(Y_test,Y_example).sum()
accuracy_percentage = (1.0*match/test_size)*100
print "running the net with 9 layers " 
print  match 
print test_size 
print accuracy_percentage # 70.6%
