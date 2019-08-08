# -*- coding: utf-8 -*-
"""Helpers to select best model."""

import implementations as imp
import visualization as visu
import numpy as np
from proj1_helpers import *
from ipywidgets import IntProgress
from IPython.display import display
from model_selection import *
import time

def calculate_accuracy(y, y_pred):
    """

    :param y: actual y
    :param y_pred: the predictions
    :return: accuracy
    """
    accuracy = np.count_nonzero(y == y_pred) / len(y) * 100
    return accuracy

def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred=imp.sigmoid(y_pred)
    y_pred = np.where(y_pred >= 0.5, 1, 0) 
    # 1,0 => 1,-1
    y_pred = 2 * y_pred
    y_pred = y_pred - 1
    return y_pred

def build_poly(x, degree):
    """Create extended feature matrix formed by applying the polynomial basis functions to all input data.

        :param x: vector of the data samples
        :param degree: maximum degree of the polynomial basis
        :return: extended feature matrix
    """
    ext_matrix = np.ones((len(x), 1))
    for deg in range(1, degree + 1) :
        ext_matrix = np.c_[ext_matrix, np.power(x, deg)]
    return ext_matrix

def build_k_indices(y, k_fold, seed):
    """Build k indices groups for k-fold.

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param k_fold: number of folds
        :param seed: random seed
        :return: indices for k-fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR LINEAR REGRESSION
# ******************************************************
def cross_validation(y, x, k_indices, k, regression_technique, **args):
    """Cross validation helper function for linear regression techniques

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param k_indices: k indices groups for k-fold
        :param k: k'th group to select
        :param regression_technique: regression technique (least_squares, etc.)
        :param args: args for regression (ex: max_iters, gamma)
        :return: loss for train, loss for test, accuracy, weights
    """
    # Build test and training set
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    X_test = x[te_indice]
    X_train = x[tr_indice]

    # Choose the regression technique
    w, loss = regression_technique(y=y_train, tx=X_train, **args)
    # calculate the loss for train and test data
    loss_tr = imp.calculate_rmse(loss)
    loss_te = imp.calculate_rmse(imp.compute_loss(y_test, X_test, w))
    accuracy = calculate_accuracy(y_test, predict_labels(w, X_test))
    return loss_tr, loss_te, accuracy, w

def cross_validation_demo(y, x, regression_technique, **args):
    """Cross validation function for linear regression techniques

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param regression_technique:
        :param args: args for regression (ex: max_iters, gamma)
    """

    seed = 12
    k_fold = 5
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    min_rmse_train = []
    min_rmse_test = []
    accuracies = []
    weights = []
    for k in range(k_fold):
        loss_tr, loss_te, accuracy_tmp, w = cross_validation(y, x, k_indices, k, regression_technique, **args)
        min_rmse_train.append(loss_tr)
        min_rmse_test.append(loss_te)
        accuracies.append(accuracy_tmp)
        weights.append(w)

    rmse_test = np.mean(min_rmse_test)
    accuracy = np.mean(accuracies)
    w = np.mean(weights, axis=0)
    print("RMSE test: ", rmse_test)
    print("Accuracy: ", accuracy)
    visu.cross_validation_visualization(np.arange(k_fold), min_rmse_train, min_rmse_test)
    return w


def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """Cross validation helper function for ridge regression techniques

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param k_indices: k indices groups for k-fold
        :param k: k'th group to select
        :param lambda_: regularization factor (penalty factor)
        :param degree: maximum degree of the polynomial basis
        :return: loss for train, loss for test, weights
    """
    # Build test and training set
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    X_test = x[te_indice]
    X_train = x[tr_indice]

    # form data with polynomial degree
    tx_train = build_poly(X_train, degree)
    tx_test = build_poly(X_test, degree)

    # ridge regression
    w, loss = imp.ridge_regression(y_train, tx_train, lambda_)

    # calculate the loss for train and test data
    loss_train = imp.calculate_rmse(loss)
    loss_test = imp.calculate_rmse(imp.compute_loss(y_test, tx_test, w))
    accuracy = calculate_accuracy(y_test, predict_labels(w, tx_test))
    return loss_train, loss_test, accuracy, w


def best_model_ridge(y, x, k_fold, degrees, lambdas, seed=56):
    """Calculate best degree and best lambda

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param k_fold: number of folds
        :param degrees:
        :param lambdas: lambdas to test
        :param seed: random seed
        :return: best degree and best lambda for ridge regression
    """
    k_indices = build_k_indices(y, k_fold, seed)
    best_lambdas = []
    best_rmses = []
    best_accuracies = []
    for degree in degrees:
        rmse_train = []
        rmse_test = []
        accuracies = []
        for lambda_ in lambdas:
            rmse_train_lambda = []
            rmse_test_lambda = []
            accuracies_lambda = []
            for k in range(k_fold):
                loss_train, loss_test, accuracy_tmp, w = cross_validation_ridge(y, x, k_indices, k, lambda_, degree)
                rmse_train_lambda.append(loss_train)
                rmse_test_lambda.append(loss_test)
                accuracies_lambda.append(accuracy_tmp)

            rmse_train.append(np.mean(rmse_train_lambda))
            rmse_test.append(np.mean(rmse_test_lambda))
            accuracies.append(np.mean(accuracies_lambda))


        ind_lambda_opt = np.argmin(rmse_test)
        best_lamda_tmp = lambdas[ind_lambda_opt]
        best_rmse_tmp = rmse_test[ind_lambda_opt]
        best_lambdas.append(best_lamda_tmp)
        best_rmses.append(best_rmse_tmp)
        best_accuracies.append(accuracies[ind_lambda_opt])
        visu.cross_validation_visualization_ridge(lambdas, rmse_train, rmse_test, degree, best_lamda_tmp, best_rmse_tmp)
        print(best_lamda_tmp, best_rmse_tmp)

    ind_best_degree = np.argmin(best_rmses)
    best_lambda = best_lambdas[ind_best_degree]
    best_degree = degrees[ind_best_degree]
    accuracy = best_accuracies[ind_best_degree]
    print("Accuracy: ", accuracy)
    return best_degree, best_lambda

# ******************************************************
# CROSS VALIDATION FUNCTIONS FOR REGULARIZED LOGISTIC REGRESSION
# ******************************************************

def best_model_logistic(y, x, k_fold, degree, lambda_, seed):
    """Calculate best degree and best lambda

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param x: vector of the data samples
        :param k_fold: number of folds
        :param degrees:
        :param lambda_: lambdas to test
        :param seed: random seed
        :return: best degree, best lambda and corresponding accuracy for logistic regression
    """
    y = np.where(y==-1, 0, y)
    best_degree = 0
    best_lambda = 0
    best_accuracy = 0
    for deg in degree:
        for lam in lambda_:
            _, accuracy, w = cross_reg_logistic_regression(y, x, degree=deg, k_fold=k_fold, 
                    lambda_= lam, max_iters=300, gamma=-1, batch=30)
            print(np.mean(accuracy))
            if np.mean(accuracy) > best_accuracy:
                best_accuracy = np.mean(accuracy)
                best_lambda = lam
                best_degree = deg
    print('best deg : {},  best lambda : {}, best accuracy : {}'.format(best_degree, best_lambda, best_accuracy))
    return best_degree, best_lambda, best_accuracy

def reg_logistic_regression(y, X, k_indices, k, degree, lam, max_iters, initial_w, gamma, batch, reg):
    """function for ridge regression

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param X: vector of the data samples
        :param k_indices: k indices groups for k-fold
        :param k: k'th group to select
        :param degree: maximum degree of the polynomial basis
        :param lam: regularization factor (penalty factor)
        :param max_iters: number of steps for the gradient descent
        :param initial_w: initial weights
        :param gamma: gradient descent factor, (1/t, for t iterations. adaptive gradient descent factor) if gamma == -1 
        :param batch: batch size
        :param reg: if True, regularized logistic regression, else, normal logistic regression
        :return: loss for train, accuracy for test, weights.
    """
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    X_test = X[te_indice]
    X_train = X[tr_indice]
    tx_training = build_poly(X_train, degree)
    tx_test = build_poly(X_test, degree)
    
    w = initial_w
    
    progressif = 0
    if gamma == -1:
        progressif = 1
    
    for n_iter in range(max_iters):
        gamma = progressif*(1/(n_iter+1)) + (1-progressif)*gamma #formula to determine gamma
        for y_b, tx_b in imp.batch_iter(y_train, tx_training, batch_size=batch, num_batches=1):
            if reg==True:
                w, loss_tr = imp.learning_by_penalized_gradient_logistic(y_b, tx_b, w, gamma, lam)
            else:
                w, loss_tr = imp.learning_by_gradient_descent_logistic(y_b, tx_b, w, gamma)
    accuracy_te = np.sum(np.where(np.abs(y_test - imp.sigmoid(tx_test @ w)) < 0.5, 1, 0)) / len(y_test)
    return loss_tr, accuracy_te, w

def cross_reg_logistic_regression(y, X, degree, k_fold, lambda_, max_iters,
                                  gamma = -1,  batch = 50, reg = True):
    """Cross validation function for logistic regression techniques

        :param y: outpus/labels, numpy array (-1 = background and 1 = signal)
        :param X: vector of the data samples
        :param degree: maximum degree of the polynomial basis
        :param k_fold: number of groups for the data
        :param lambda_: regularization factor (penalty factor)
        :param max_iters: number of steps for the gradient descent
        :param gamma: gradient descent factor, (1/t, for t iterations. adaptive gradient descent factor) if gamma == -1 
        :param batch: batch size
        :param reg: if True, regularized logistic regression, else, normal logistic regression
        :return: loss for train, accuracy for test, weights. Computed in mean over all k_folds
    """
    y = np.where(y==-1, 0, y)
    w = np.ones(X.shape[1]*degree + 1)
    seed=13
    k_indices = build_k_indices(y, k_fold, seed)
   
    list_accuracy_te = []
    list_loss_tr = []
    list_w = []
    for k in range(k_fold):
        loss_tr, accuracy_te, w = reg_logistic_regression(y, X, k_indices, k, degree, lambda_, max_iters, w, gamma, batch, reg)
        list_accuracy_te.append(accuracy_te)
        list_loss_tr.append(loss_tr)
        list_w.append(w)
    return np.mean(list_loss_tr), np.mean(list_accuracy_te), np.mean(list_w, axis=0)
