# -*- coding: utf-8 -*-
"""additional visualization functions used for project 1."""
import numpy as np
import matplotlib.pyplot as plt


def get_jet_masks(x):
    """Separate the data into subsets depending on pri_jet_num

        :param x: data
        :return: subsets that have the same pri_jet_num
    """
    dictionary = {0: x[:, 22] == 0, 1: x[:, 22] == 1, 2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)}
    return dictionary


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.ylabel("rmse")
    plt.ylim([0.0, 2])
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()


def cross_validation_visualization_ridge(lambds, mse_tr, mse_te, degree=0, marker_x=None, marker_y=None):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation for degree " + str(degree))
    plt.legend(loc=2)
    plt.grid(True)
    plt.plot(marker_x, marker_y, marker='*', color='g', markersize=10)
    plt.show()


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")