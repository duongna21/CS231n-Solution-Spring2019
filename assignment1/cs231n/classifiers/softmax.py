from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute loss

    N, D = X.shape
    C = W.shape[1]

    for i in range(N):
        score = np.dot(X[i], W)
        score -= np.max(score)
        p = np.exp(score)/np.sum(np.exp(score))

        for j in range(C):
            dW[:, j] += X[i] * (p[j] - (j==y[i]))

        loss_i = -np.log(p[y[i]])
        loss += loss_i
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
    dW /= N
    dW += reg * W

    # compute gradient



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = len(y)

    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)

    f_y = scores[np.arange(num_train), y]
    exp_f_y = np.exp(f_y)

    loss = -np.sum(np.log(exp_f_y/np.sum(exp_scores, axis=1)))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    ind = np.zeros_like(scores)
    ind[np.arange(num_train), y] = 1

    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    dW = np.dot(X.T, softmax - ind)

    dW /= num_train
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
