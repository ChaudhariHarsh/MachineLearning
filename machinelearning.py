"""      """


import numpy as np
import matplotlib.pyplot as plt
import math

def initialize_parameters(n_x,method):
    ##This function is for initialize All parameter
    
    parameters = {}
    if method == "Linear":
        W = np.zeros((1,n_x))
        b = 0
        parameters = {"W": W,
                      "b": b}
    elif method == "Logistic":
        W = np.zeros((1,n_x))
        b = 0
        parameters = {"W": W,
                      "b": b}
    else :
        print "Error : Define method in initialize parameter"
        
    
    return W, b
#parameters = initialize_parameters(5,method="Logistic")
#print parameters

def feature_normalization(X):
    (row, col) = X.shape
    print X.shape
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
        error = np.sum(cost /(m),axis=1)
    elif method == "Logistic":
        Z = np.dot(W,X) + b 
        hyponthsis_function = 1/( 1 + np.exp(-Z))
        cost = - np.dot(Y,np.log(hyponthsis_function.T)) - np.dot(1 - Y,np.log(1 - hyponthsis_function.T)) 
        error = np.sum(cost /(m),axis=1)
    else:
        print "Error In Cost Function : No method Found"
    print error
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
    X = np.zeros((j,k+2),dtype=float)
    Y = np.zeros((j,1),dtype=float)
    Xv = np.zeros((j,k+2),dtype=float)
    Yv = np.zeros((j,1),dtype=float)
    for i in range(j):
        X[i,0]= 1
        X[i,1]=Z[i,0]
        X[i,2]=Z[i,0]**(1/2.0)
        X[i,3]=Z[i,1]
        X[i,4]=Z[i,1]**(1/2.0)
        Y[i,0]=Z[i,2]
        Xv[i,0]= 1
        Xv[i,1]=Zv[i,0]
        Xv[i,2]=Zv[i,0]**(1/2.0)
        Xv[i,3]=Zv[i,1]
        Xv[i,4]=Zv[i,1]**(1/2.0)
        Yv[i,0]=Zv[i,2]
    X = np.matrix(X)
    Xv = np.matrix(Xv)

    X = X.T ## Make this X with shape of (5,47)
    Y = Y.T ## Make this Y with shape of (1,47)
    Xv = np.matrix(Xv)
    Xv = Xv.T ## Make this X with shape of (5,47)
    Yv = Yv.T ## Make this Y with shape of (1,47)
    hyponthsis_function, error, W, b = linear_regression(X, Y, itertions=1000, learning_rate=0.5, method="Linear")
    ##h = hyponthsis_function.T
    #print Y[0,5], hyponthsis_function[0,5]


    x = Xv[1,:].T
    x = np.array(x)
    x = x.flatten()
    y = Yv.T
    y = np.array(y)
    y = y.flatten()
    hf, error, W, b = linear_regression(Xv, Yv, itertions=100, learning_rate=0.5, method="Linear")
    hf = np.array(hf)
    hf = hf.flatten()
    #print X,X.shape,hyponthsis_function.shape
    A = visualization(x, y, hf)

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
    X = X.T ## Make this X with shape of (5,47)
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
    print sum((Y_prediction.T-Y.T)**2)
    f, ax = plt.subplots(figsize=(10, 7))
    X = feature_normalization(X)
    x1 = np.array(X[0,:]).flatten()
    x2 = np.array(X[2,:]).flatten()
    y = np.array(Y[0,:]).flatten()
    y2 = 2-y
    hf1 = np.array(hyponthsis_function[0,:]).flatten() - 0.5
    hf2 = 0.5 - np.array(hyponthsis_function[0,:]).flatten()
    ax.scatter(x1, x2,hf1*100,marker='+',c="y")
    ax.scatter(x1, x2,hf2*100,marker='o',c="g")
    ax.scatter(x1[c], x2[c],50,marker='x',c="r")

    ax.set(aspect="equal",
           xlim=(-0.2, 1.2), ylim=(1.2, -0.2),
           xlabel="$X_1$", ylabel="$X_2$")
    f.show()

logisticCall()
