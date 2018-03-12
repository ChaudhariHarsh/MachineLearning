""" ------This is Machine Learning Function File-------
    Developed by : Harsh Chaudhari
    Discription : In this file I have developed simple
                  Implementation for Linear regression
                  and Logistic Regression. In linear
                  regression example I made model for
                  Houcing price prediction and in Logistic
                  regression example i developed model
                  for student admitted to exam or not"""


import numpy as np
import matplotlib.pyplot as plt
import math


def initialize_parameters(n_x,method):
    ##This function is for initialize All parameter    
    parameters = {}
    if method == "Linear":
        W = np.zeros((1,n_x))
        b = 0
    elif method == "Logistic":
        W = np.zeros((1,n_x))
        b = 0
    else :
        print "Error : Define method in initialize parameter"
    return W, b
#parameters = initialize_parameters(5,method="Logistic")
#print parameters

def feature_normalization(X):
    (row, col) = X.shape
    for f in range(row):
        
        X[f,:] = (X[f,:]- min(X[f,:].T))/(max(X[f,:].T)- min(X[f,:].T))
    
    assert(X.shape==(row,col)),"Error in size match : feature_normalization"
    
    return X


def cost_function(X, Y, W, b, method):
    ## where X shape is (input_size, no_examples)
    (n,m) = X.shape
    if method == "Linear":
        hyponthsis_function = np.dot(W,X) + b 
        cost = np.square(hyponthsis_function - Y)
        error = np.sum(cost /(2*m),axis=1)
    elif method == "Logistic":
        Z = np.dot(W,X) + b 
        hyponthsis_function = 1/( 1 + np.exp(-Z))
        cost = - np.dot(Y,np.log(hyponthsis_function.T)) - np.dot(1 - Y,np.log(1 - hyponthsis_function.T)) 
        error = np.sum(cost /(m),axis=1)
    else:
        print "Error In Cost Function : No method Found"
    assert(hyponthsis_function.shape == (1,m)),"Assertation Error in hyponthsis_function at cost function "
    
    return hyponthsis_function, cost, error


def gradient_descent(X, Y, W, b, itertions, learning_rate, method):
    (n,m) = X.shape
    for iteration in range(itertions):
        if method == "Linear":
            dJ = np.dot(W,X) + b - Y
            db = np.sum(dJ, axis=0)/(2*m)
            dW = np.sum(np.dot(dJ,X.T), axis=0)/(2*m)
        elif method == "Logistic":
            Z = np.dot(W,X) + b
            hf = 1/(1+ np.exp(-Z))
            dJ = hf - Y
            db = np.sum(dJ, axis=1)/m
            dW = np.sum(np.dot(dJ,X.T), axis=0)/m
            
        else:
            print "Error in gradient descent: No method Found"
        #print hf.T, Y.T
        W = W - learning_rate * dW
        b = b - learning_rate * db
        
    assert(dJ.shape == (W.shape[0],X.shape[1]))
    assert(db.shape == b.shape)
    assert(dW.shape == W.shape)
    return W,b


def visualization(x, y, hf):
    fig, handle = plt.subplots()
    handle.plot(x, y, "yo", x, hf, "--k")
    #handle.plot(x, hf, color='red')
    #handle.scatter(x, y)
    fig.show()
    return None


def linear_regression(X, Y, itertions, learning_rate, method = "Linear"):
    
    (row, col) = X.shape

    (W, b) = initialize_parameters(row,method)

    X = feature_normalization(X)

    (W,b) = gradient_descent(X, Y, W, b, itertions, learning_rate, method)

    hyponthsis_function, cost, error = cost_function(X, Y, W, b, method)
    
    return hyponthsis_function, error, W, b
          

def logistic_regression(X, Y, itertions, learning_rate, method):
    
    (row, col) = X.shape

    (W, b) = initialize_parameters(row,method)

    X = feature_normalization(X)

    (W,b) = gradient_descent(X, Y, W, b, itertions, learning_rate, method)

    hyponthsis_function, cost, error = cost_function(X, Y, W, b, method)
    
    return hyponthsis_function, error, W, b

