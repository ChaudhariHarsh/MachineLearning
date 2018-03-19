# ```MachineLearning :``` 

- In this repository I will develop functions for linear regression, logistic regression, neural network. This all function will be without regulization part, so we may endup with overfit model or we can say "high variance".

## ```Linear Regression Function :```
- In this section, we will develop linear regression function that can handle to predict response with multiple features. For more better result I will use more feature with square and root combinations so that we can fit non linear learning curve.

- As per general naming convention for machine learning, here we presents common terminalogy.

```
X = Features,
Y = Resposes,
W = weights - Learning Parameter,
b = bias - Learning Parameter,
α = Learning Rate,
m = Training set size
lambda = regularization parameter
```

### Hyponthsis function :

> h(x) = b + w(1) * x(1) + w(2) * x(2) + w(3) * x(3) ...

- As we know `small letter x, w ` means scaller where as `big letter X, W ` means vector or matrix. This is very importent when we are implementing because it helps us prevent misunderstanding in complex coding.

### Cost Functoin :

> cost = ( h(x) - y )^2  + lambda * np.sum(W**2)

- we use squared error cost function in this linear regression example because its most commonly used for such task.

### Gradient Decent :

> grad J = 2 * (h(x) - y) 

- This is gradient of cost function with respect to bias(b), for gradient for cost with respect to weight(w) we can derive as follow.

> grad J = 2 * x * (h(x) - y) + lambda * W

### Weight Update : 

- Updating Bias with learning rate α,

> b := b * (1 - lambda/m) - (α / 2 * m) (**∑** (h(x) - y))

- Updating weight W with learning rate α,

> w[ i ] := w[ i ] * (1 - lambda/m) - (α / 2 * m) (**∑** (h(x) - y)* x( i ))

### Visualization :

- Here we are visualizing trained data with learning curve using **iteration = 1000** and **learning_rate = 0.01**.

![Alt text](https://github.com/ChaudhariHarsh/MachineLearning/blob/master/LinearRe.png)

- In this figure, blue dots represents data set with **size of house** in *X-axis* and **price of house** in *Y-axis*. *Red line* represents for *learning curve* of trained model. This way we can see that how model being trained.


## ```Logistic Regression Function :```

- In this section, we will develop logistic regression function that can handle to classify with multiple features. For more better result I will use more feature with square and root combinations so that we can fit non-linear decision boundary.


### Hyponthsis function :

> z(x) = b + w(1) * x(1) + w(2) * x(2) + w(3) * x(3) ...

> h(x) = 1/(1 + exp(z(x)))

### Cost Functoin :

> cost = - y * log(h(x)) - (1 - y) * log(1-h(x)) + lambda * np.sum(W**2)

- we use sigmoid function in this logistic regression example.

### Gradient Decent :

> grad J = (cost - y)

- This is gradient of cost function with respect to bias(b), for gradient for cost with respect to weight(w) we can derive as follow.

> grad J = 2 * x * (cost - y) + lambda * W

### Weight Update : 

> b := b * (1 - lambda/m) - (α / 2 * m) (**∑** grad J)

> w[ i ] := w[ i ] * (1 - lambda/m) - (α / 2 * m) (**∑** grad J * x( i ))

### Visualization :

- Here we are visualizing trained data with learning curve using **iteration = 1000** and **learning_rate = 0.01** 

![Alt text](https://github.com/ChaudhariHarsh/MachineLearning/blob/master/LogRe.png)

- In this figure, green dots represents correctly classified for **student admitted** to exam where as yello dot represents for correctly classified for **student not admitted** to exam. Red dot represents for not **currectly classified**. This way we can see that how logistic regression model being trained.

- Or We can see as below,

![Alt text](https://github.com/ChaudhariHarsh/MachineLearning/blob/master/LogisticRe.png)


## ```Neural Netowrk implementation :``` 
- I have sucessufully implemented neural network for above example of logistic regression. In here i used same data file.

### Visualization :
- this is exectly same results but different colour.

![Alt text](https://github.com/ChaudhariHarsh/MachineLearning/blob/master/neuralnet.png)

- This is visualizarion figure with normalization implementation.

![Alt text](https://github.com/ChaudhariHarsh/Housing-Price-prediction/blob/master/LinearReg.png)
