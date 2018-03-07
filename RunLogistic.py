
import numpy as np
import matplotlib.pyplot as plt
import math
from supervisedlearning import *


def logisticCall():
    Z = []
    file = open("ex2data1.txt", "r") 
    for line in file:
        line = line.split(',')
        Z.append(line)
    
    Z = np.matrix(Z, dtype=float)
    (j, k) = Z.shape
    X = np.zeros((j,k),dtype=float)
    Y = np.zeros((j,1),dtype=float)
    Y_prediction = np.zeros((1,j),dtype=float)
    for i in range(j):
        X[i,0]= Z[i,0]
        X[i,1]= Z[i,0]**(1/2.0)
        X[i,2]= Z[i,1]
        Y[i,0]=Z[i,2]
    X = np.matrix(X)
    X = X.T ## Make this X with shape of (4,47)
    Y = Y.T ## Make this Y with shape of (1,47)
    #print X
    hyponthsis_function, error, W, b = logistic_regression(X, Y, itertions=10000, learning_rate=0.01, method="Logistic")
    for i in range(hyponthsis_function.shape[1]):
        if (hyponthsis_function[0,i] > 0.5):
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
        if Y_prediction[0,i]!=Y[0,i]:
            print "Error at :",i
    zz = (Y_prediction.T-Y.T)**2
    c = np.where(zz == 1)[0]
    print "Total Number of Error : ",sum(zz)
    f, ax = plt.subplots(figsize=(10, 7))
    X = feature_normalization(X)
    x1 = np.array(X[0,:]).flatten()
    x2 = np.array(X[2,:]).flatten()
    y = np.array(Y[0,:]).flatten()
    y2 = 1-y
    hf1 = np.array(hyponthsis_function[0,:]).flatten() - 0.5
    hf2 = 0.5 - np.array(hyponthsis_function[0,:]).flatten()
    ax.scatter(x1, x2,y*50,marker='+',c="b")
    ax.scatter(x1, x2,y2*50,marker='o',c="g")
    ax.scatter(x1[c], x2[c],50,marker='x',c="r")

    ax.set(aspect="equal",
           xlim=(-.2, 1.2), ylim=(-0.2, 1.2),
           xlabel="$X_1$", ylabel="$X_2$")
    f.show()

logisticCall()
