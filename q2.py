"""
Created on Tue Sep 12 20:39:09 2017
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    # compute the distance of test_datum to every examples in test vector
    test_datum = test_datum.reshape(test_datum.shape[0],1)
    distance = l2(test_datum.T,x_train) # dim 1 x N_train
    # compute log(a)
    log_num = -distance/(2*(tau**2))
    log_denom = logsumexp(log_num)
    # compute a
    a = np.exp(log_num-log_denom)
    # compute w_star
    A= np.diagflat(a)
    w_star = np.linalg.solve(np.dot(np.dot(x_train.T,A),x_train)+lam*np.identity(x_train.shape[1]),np.dot(np.dot(x_train.T,A),y_train))
    y_hat = np.dot(test_datum.T,w_star)

    return y_hat



def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    # Split the training set x_train and validation set x_valid
    N_train = int(x.shape[0]*(1-val_frac))
    np.random.seed(0)
    idx = np.random.permutation(range(x.shape[0]))
    x_train = x[idx[0:N_train],:]
    y_train = y[idx[0:N_train]]
    x_valid = x[idx[N_train:],:]
    y_valid = y[idx[N_train:]]
    # construct the vectors to store the train loss and validation loss:
    train_losses= np.array([])
    test_losses = np.array([])
    # compute losses for a single tau
    for tau in taus:
        train_loss = 0.0
        test_loss = 0.0
        for i in range(N_train):
            test_datum = x_train[i,:]
            y_hat = LRLS(test_datum,x_train,y_train,tau)
            train_loss += (y_hat-y_train[i])**2
        train_loss = train_loss/N_train
        train_losses = np.append(train_losses,train_loss)
        for j in range(len(y_valid)):
            test_datum = x_valid[j,:]
            y_hat = LRLS(test_datum,x_train,y_train,tau)
            test_loss += (y_hat-y_valid[j])**2
        test_loss = test_loss/len(y_valid)
        test_losses = np.append(test_losses,test_loss)

    return train_losses,test_losses
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(train_losses,label='Train')
    plt.semilogx(test_losses, label='Test')
    plt.legend()
    plt.show()