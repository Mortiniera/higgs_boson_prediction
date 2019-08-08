# -*- coding: utf-8 -*-
"""additional helper functions used for project 1."""

from proj1_helpers import load_csv_data
import numpy as np


def build_model_data(y_feature, x_feature):
    """Form (y,tX) to get regression data in matrix form."""
    y = y_feature
    x = x_feature
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def standardize(x):
    """Standardize the original data set.
    
        :param x: data
        :return: standardized data
    """""
    x_std = np.std(x, axis=0)
    x_mean = np.mean(x, axis=0)
    return (x - x_mean)/x_std, x_mean, x_std


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization.

        :param x: data
        :param mean_x: mean of data
        :param std_x: standard deviation of data
        :return: destandardized data
    """
    x = x * std_x
    x = x + mean_x
    return x


def get_jet_masks(x):
    """Separate the data into subsets depending on pri_jet_num

        :param x: data
        :return: subsets that have the same pri_jet_num
    """
    dictionary = {0: x[:, 22] == 0, 1: x[:, 22] == 1, 2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)}
    return dictionary

def inv_sinh_f(X) :
    """Apply arcsinh. Use log_f(X) to get better results.

        :param X: data
        :return: data with arcsinh applied on columns
    """

    x_inv = X

    for i in range(len(X)):
        x_inv_sinh_cols = np.arcsinh(X[i][:, :])
        x_inv[i] = np.hstack((X[i], x_inv_sinh_cols))

    return x_inv


def log_f(X) :
    """Apply log on positive columns

            :param X: data
            :return: data with log applied on positive columns
    """

    x_inv = X

    for i in range(len(X)):
        strictly_positive_cols = [j for j in range(len(X[i][0])) if (X[i][:, j] > 0).all()]
        x_log_cols = np.log(X[i][:, strictly_positive_cols])
        x_inv[i] = np.hstack((X[i], x_log_cols))

    return x_inv


def process_data(path, inv_log=False):
    """Process the data before using it doing some engineering featuring

        :param path: path of the dataset
        :param inv_log: apply log on the positive columns of the dataset
        :return: y, processed data, masks based on pri_jet_num, ids
    """
    y, X, ids = load_csv_data(path)

    dict_mask_jets_train = get_jet_masks(X)

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
        new_X = log_f(new_X)

    for i in range(len(dict_mask_jets_train)):
        new_X[i], x_mean, x_std = standardize(new_X[i])

    return y, new_X, dict_mask_jets_train, ids



