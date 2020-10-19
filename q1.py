import numpy as np

def train_huber(X, targets, delta):
    lr = 0.01  ## learning rate

    '''Input : X, targets [data and outcome], delta value
       Output : w,b 
    '''

    ## initialize the values of w and b to be zero
    w = np.zeros(X.shape[1])
    b = 0

    ## Run gradient descent
    for i in range(8000):
        prediction = np.dot(X,w)+b
        difference = prediction-targets
        ## Calculate the gradient of our function
        gradient = np.copy(difference)
        gradient = np.where(gradient > delta, delta, gradient)
        gradient = np.where(gradient < -delta, -delta, gradient)
        ## Find the gradient correspoding to each parameters
        w_gradient = np.dot(X.T,gradient)/X.shape[0]
        b_gradient = gradient.sum()/X.shape[0]

        w -= lr * w_gradient
        b -= lr * b_gradient

    return w,b


# Ground truth weights and bias used to generate toy data
w_gt = [1, 2, 3, 4]
b_gt = 5

# Generate 100 training examples with 4 features
X = np.random.randn(100, 4)
targets = np.dot(X, w_gt) + b_gt

# Gradient descent
w, b = train_huber(X, targets, delta=2)

