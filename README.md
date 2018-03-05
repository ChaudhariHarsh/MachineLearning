# MachineLearning : 
In this repository I will develop functions for linear regression, logistic regression, neural network.

## Linear Regression Function :
In this section, we will develop linear regression function that can handle to predict response with multiple features. For more better result I will use more feature with square and root combinations so that we can fit non linear learning curve.

As per general naming convention for machine learning, here we presents common terminalogy.
```
X = Features,
Y = Resposes,
W = weights - Learning Parameter,
b = bias - Learning Parameter,
α = Learning Rate,
m = Training set size
```

### Hyponthsis function :

> h(x) = b + w(1) * x(1) + w(2) * x(2) + w(3) * x(3) ...

As we know `small letter x, w ` means scaller where as `big letter X, W ` means vector or matrix. This is very importent when we are implementing because it helps us prevent misunderstanding in complex coding.

### Cost Functoin :

> cost = ( h(x) - y )**2

we use squared error cost function in this linear regression example.

### Gradient Decent :

> grad J = 2 * (h(x) - y)

This is gradient of cost function with respect to bias(b), for gradient for cost with respect to weight(w) we can derive as follow.

> grad J = 2 * x * (h(x) - y)

### Weight Update : 

> b := b - (α / 2 * m) (**∑** (h(x) - y))

> w[ i ] := w[ i ] - (α / 2 * m) (**∑** (h(x) - y)* x( i ))

### Visualization :

Here we are visualizing trained data with learning curve using **iteration = 1000** and **learning_rate = 0.01**.

- [ ] This is still under development.
