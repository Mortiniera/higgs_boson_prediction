# -*- coding: utf-8 -*-
"""Creates the prediction independantly of the project
(some methods are rewritten here to have all in the same file)."""

import sys
sys.path.insert(0, 'scripts')
import numpy as np
import implementations as imp
import model_selection as modselection
import proj1_helpers as helpers
import helpers_us as helpers_us


def process_data(path, inv_log=False):
    """Process the data before using it doing some engineering featuring

        :param path: path of the dataset
        :param inv_log: apply log on the positive columns of the dataset
        :return: y, processed data, masks based on pri_jet_num, ids
    """
    y, X, ids = helpers.load_csv_data(path)

    dict_mask_jets_train = helpers_us.get_jet_masks(X)

    new_X = []

    for i in range(len(dict_mask_jets_train)):
        new_X.append(np.delete(X[dict_mask_jets_train[i]], [22, 29], axis=1))

    for i in range(len(dict_mask_jets_train)):
        undefined_columns = [j for j in range(len(new_X[i][0])) if (new_X[i][:, j] < -900).all()]
        new_X[i] = np.delete(new_X[i], undefined_columns, axis=1)

    for i in range(len(dict_mask_jets_train)):
        for j in range(len(new_X[i][0])):
            col = new_X[i][:, j]
            np.where(col < -900)
            m = np.mean(col[col >= -900])  # compute mean of the right columns
            col[np.where(col < -900)] = m
            new_X[i][:, j] = col

    if inv_log:
        new_X = helpers_us.log_f(new_X)

    for i in range(1, len(dict_mask_jets_train)):
        new_X[i], x_mean, x_std = helpers_us.standardize(new_X[i])

    return y, new_X, dict_mask_jets_train, ids

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
    tx_training = modselection.build_poly(X_train, degree)
    tx_test = modselection.build_poly(X_test, degree)

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
    k_indices = modselection.build_k_indices(y, k_fold, seed)

    list_accuracy_te = []
    list_loss_tr = []
    list_w = []
    for k in range(k_fold):
        loss_tr, accuracy_te, w = reg_logistic_regression(y, X, k_indices, k, degree, lambda_, max_iters, w, gamma, batch, reg)
        list_accuracy_te.append(accuracy_te)
        list_loss_tr.append(loss_tr)
        list_w.append(w)
    return np.mean(list_loss_tr), np.mean(list_accuracy_te), np.mean(list_w, axis=0)

def create_prediction():
    """Create predictions for kaggle."""

    y, X, dict_mask_jets_train, ids = helpers_us.process_data('Data/train.csv', inv_log=True)
    best_param = [[2,0.0072],[2,0.1389],[2,0.1389]] #found with the function best_model_logistic
    best_w = []

    for i in range(len(dict_mask_jets_train)):
        xi = X[i]
        yi = y[dict_mask_jets_train[i]]
        _,_,w = cross_reg_logistic_regression(yi, xi, degree = best_param[i][0], k_fold=6,
                                             lambda_= best_param[i][1], max_iters = 500, gamma = -1, batch=35)
        best_w.append(w)

    y, X, dict_mask_jets_train, ids = helpers_us.process_data('Data/test.csv', inv_log=True)
    y_pred = np.zeros(y.shape[0])
    for i in range(len(dict_mask_jets_train)):
        xi = X[i]
        xi = modselection.build_poly(xi, 2)
        y_test_pred = modselection.predict_labels_logistic(best_w[i], xi)
        y_pred[dict_mask_jets_train[i]] = y_test_pred

    helpers.create_csv_submission(ids, y_pred, "true_prediction.csv")


if __name__ == '__main__':
    create_prediction()
