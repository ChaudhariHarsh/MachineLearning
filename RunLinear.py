
import numpy as np
import matplotlib.pyplot as plt
import math
from supervisedlearning import *


def linearCall():
    Z = []
    file = open("data.txt", "r") 
    for line in file:
        line = line.split(',')
        Z.append(line)
        
    Z = np.matrix(Z, dtype=float)
    Zv = Z
    Zv = np.sort(Zv, axis=0)
    (j, k) = Z.shape
    X = np.zeros((j,k+1),dtype=float)
    Y = np.zeros((j,1),dtype=float)
    Xv = np.zeros((j,k+1),dtype=float)
    Yv = np.zeros((j,1),dtype=float)
    for i in range(j):
        X[i,0]=Z[i,0]
        X[i,1]=Z[i,0]**(1/2.0)
        X[i,2]=Z[i,1]
        X[i,3]=Z[i,1]**(1/2.0)
        Y[i,0]=Z[i,2]
        Xv[i,0]=Zv[i,0]
        Xv[i,1]=Zv[i,0]**(1/2.0)
        Xv[i,2]=Zv[i,1]
        Xv[i,3]=Zv[i,1]**(1/2.0)
        Yv[i,0]=Zv[i,2]
    X = np.matrix(X)
    Xv = np.matrix(Xv)

    X = X.T ## Make this X with shape of (4,47)
    Y = Y.T ## Make this Y with shape of (1,47)
    Xv = np.matrix(Xv)
    Xv = Xv.T ## Make this X with shape of (4,47)
    Yv = Yv.T ## Make this Y with shape of (1,47)
    hyponthsis_function, error, W, b = linear_regression(X, Y, itertions=5600, learning_rate=0.1, method="Linear")
    ##h = hyponthsis_function.T
    #print Y[0,5], hyponthsis_function[0,5]


    x = Xv[0,:].T
    x = np.array(x)
    x = x.flatten()
    y = Yv.T
    y = np.array(y)
    y = y.flatten()
    hf, error, W, b = linear_regression(Xv, Yv, itertions=5600, learning_rate=0.1, method="Linear")
    hf = np.array(hf)
    hf = hf.flatten()
    #print Y, hyponthsis_function
    A = visualization(x, y, hf)

linearCall()
