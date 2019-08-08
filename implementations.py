# -*- coding: utf-8 -*-
"""Implementations of the ML methods."""
import numpy as np

def calculate_mse(err):
    """Calculate the mean-square-error.

        :param err: error vector
        :return: mse
    """
    return 1/2*np.mean(err**2)

def calculate_rmse(err_mse):
    """Calculate the root-mean-square-error.

        :param err: mse error vector
        :return: rmse
    """
    return np.sqrt(2*err_mse)

def compute_loss(y, tx, w):
    """Calculate loss value.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param w: weights used to calculate loss
        :return: loss value
    """
    err = y - tx.dot(w)
    return calculate_mse(err)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_gradient(y, tx, w):
    """Compute the gradient.

    :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
    :param tx: standardized inputs/features augmented with the first column filled with 1's
    :param w: weights
    :return: gradient
    """
    err = y - tx.dot(w)
    gradient = -tx.T.dot(err)/len(tx)
    return gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param initial_w: initial weight vector
        :param max_iters: number of iterations for the gradient descent
        :param gamma: learning rate

        :return: (w, loss) where w is the last weight vector and loss is the corresponding loss value
    """
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
        # compute loss
        loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param initial_w: initial weight vector
        :param max_iters: number of iterations for the gradient descent
        :param gamma: learning rate
        :return: (w, loss) where w is the last weight vector and loss is the corresponding loss value
    """
    w = initial_w
    loss = 0
    batch_size = 20
    for n_iter in range(max_iters):
        # calculate gradient with the batches
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # calculate loss
            loss = compute_loss(y, tx, w)
            # compute gradient
            stoch_gradient = compute_gradient(batch_y, batch_tx, w)
            # update w by stochastic gradient
            w = w - gamma * stoch_gradient
    return w, loss

def least_squares(y, tx):
    """Least squares regression using normal equations.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :return: (w, loss) where w is the last weight vector and loss is the corresponding loss value
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param lambda_: regularization factor (penalty factor)
        :return: (w, loss) where w is the last weight vector and loss is the corresponding loss value
    """
    lambda_prime = 2 * len(tx) * lambda_
    A = tx.T.dot(tx) + lambda_prime * np.identity(tx.shape[1])
    B = tx.T.dot(y)
    w = np.linalg.solve(A, B)
    loss = compute_loss(y, tx, w)
    return w, loss

def sigmoid(t):
    """Apply vectorized sigmoid function on t.
        For more numerical precision, we chose to  calculating the sigmoid
        by two mathematically equivalent expressions.

        :param t: element
        :return: sigmoid(element)
    """
    def sig_elem(z):
        if z <= 0:
            return np.exp(z) / (np.exp(z) + 1)
        else:
            return 1 / (1 + np.exp(-z))
    return np.vectorize(sig_elem)(t)

def calculate_loss_logistic(y, tx, w):
    """Compute the cost by negative log likelihood.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param w: weights used to calculate loss
        :return: logistic loss
    """
    y = y.reshape((-1, 1))
    return np.sum(np.logaddexp(0, tx.dot(w))) - y.T.dot(tx.dot(w))

def calculate_gradient_logistic(y, tx, w):
    """Compute the gradient of loss.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param w: weights
        :return: logistic gradient
    """
    return tx.T.dot(sigmoid(tx.dot(w))-y)

def learning_by_gradient_descent_logistic(y, tx, w, gamma):
    """Do one step of gradient descent using logistic regression.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param w: weights
        :param gamma: learning rate
        :return: updated w and loss
    """
    # compute loss
    loss = calculate_loss_logistic(y, tx, w)
    # compute gradient
    gradient = calculate_gradient_logistic(y, tx, w)
    # update w by gradient
    w = w - gamma * gradient
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param initial_w: initial weight vector
        :param max_iters: number of iterations
        :param gamma: learning rate
        :return: (w, loss) where w is the last weight vector and loss is the corresponding loss value
    """
    max_iters = 300
    w = np.ones(tx.shape[1])

    for n_iter in range(max_iters):
        gamma = 1 / (n_iter + 1)
        for y_b, tx_b in batch_iter(y, tx, batch_size=20, num_batches=1):
            w, loss = learning_by_gradient_descent_logistic(y_b, tx_b, w, gamma)
    return w, loss

def penalized_logistic_regression(y, tx, w, lambda_):
    """Calculate gradient and loss for penalized logistic regression

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param w: weights
        :param lambda_: regularization factor (penalty factor)
        :return: gradient and loss
    """
    gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    loss = calculate_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return gradient, loss

def learning_by_penalized_gradient_logistic(y, tx, w, gamma, lambda_):
    """Do one step of gradient descent, using the penalized logistic regression.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param w: actual weights
        :param gamma: learning rate
        :param lambda_: regularization factor (penalty factor)
        :return: updated weight and loss
    """
    gradient, loss = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return w, loss

def reg_logistic_regression(y, tx, lambda_):
    """Regularized logistic regression using SGD

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param tx: standardized inputs/features augmented with the first column filled with 1's
        :param lambda_: regularization factor (penalty factor)
        :param initial_w: initial weight vector
        :param max_iters: number of iterations
        :param gamma: learning rate
        :return: (w, loss) where w is the last weight vector and loss is the corresponding loss value
    """
    max_iters = 500
    w = np.ones(tx.shape[1])
    for n_iter in range(max_iters):
        gamma = 1/(n_iter+1)
        for y_b, tx_b in batch_iter(y, tx, batch_size=30, num_batches=1):
            w, loss = learning_by_penalized_gradient_logistic(y_b, tx_b, w, gamma, lambda_)
    return w, loss
