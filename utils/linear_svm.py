# linear_svm.py ---
#
# Filename: linear_svm.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 20:45:06 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

import numpy as np


def model_predict(W, b, x):
    """Prediction function.

    Note that this function can be merged with model_loss, but is written
    as a separate function simply to enhance readibility.

    Parameters
    ----------
    W : ndarray
        The weight parameters of the linear classifier. D x C, where C is the
        number of classes, and D is the dimenstion of input data.

    b : ndarray
        The bias parameters of the linear classifier. C, where C is the number
        of classes.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    Returns
    -------
    pred : ndarray
        Predictions from the model. N numbers where each number corresponds to
        a class.

    """

    # Scores for all class (N, 10)
    s_all = np.matmul(x, W) + b
    # Predections to use later
    pred = np.argmax(s_all, axis=1)

    return pred


def model_loss(W, b, x, y):
    """Loss function.

    Also computes the individual per-class losses to be used for gradient
    computation, as well as the predicted labels.

    Parameters
    ----------
    W : ndarray
        The weight parameters of the linear classifier. D x C, where C is the
        number of classes, and D is the dimenstion of input data.

    b : ndarray
        The bias parameters of the linear classifier. C, where C is the number
        of classes.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    loss : float
        The average loss coming from this model. In the lecture slides,
        represented as \sum_i L_i.

    loss_c : ndarray
        The individual losses per sample and class, or any other value that
        needs to be reused when computing gradients.

    pred : ndarray
        Predictions from the model. N numbers where each number corresponds to
        a class.

    """

    # Scores for all class (N, 10)
    s_all = np.matmul(x, W) + b
    # Predections to use later
    pred = np.argmax(s_all, axis=1)
    # Score for the correct class (N, )
    s_y = s_all[np.arange(len(y)), y]
    # Make Nx1 to sub from s_all
    s_y = np.reshape(s_y, (-1, 1))
    # Loss per class (including the correct one)
    loss_c = np.maximum(0, s_all - s_y + 1)
    # Compute loss by averaging of samples, summing over classes. We subtract 1
    # after the sum, since the correct label always returns 1 in terms of the
    # per-class loss, and should be excluded from the final loss
    loss = np.mean(np.sum(loss_c, axis=1) - 1, axis=0)

    return loss, loss_c, pred


def model_grad(loss_c, x, y):
    """Gradient.

    Parameters
    ----------
    loss_c : ndarray
        The individual losses per sample and class, or any other value that
        needs to be reused when computing gradients.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    dW : ndarray
        Gradient associated with W. Should be the same shape (DxC) as W.

    db : ndarray
        Gradient associated with b. Should be the same shape (C) as b.

    """

    # loss_c : (N, C)
    # x : (N, D)
    # y : (N, )

    N, C = loss_c.shape
    D = x.shape[1]

    # Get number of penalized classes per sample (K in slides)
    K = np.sum(loss_c > 0, axis=1) - 1  # (N, )

    # ----------------------------------------
    # For the W (N, D, C)
    #
    # First, create a mask where we have samples within margin, i.e. where loss
    # is generated (second case). We also prepare it for broadcasting
    mask = np.reshape(loss_c > 0, (N, 1, C)).astype(float)
    # Then, the gradients that go here is simply the samples where the mask
    # survives.
    dW = np.reshape(x, (N, D, 1)) * mask
    # Now we fill the k=y part -- create weighted xs
    dW1 = np.reshape(-K, (N, 1)) * x  # (N, D)
    # Fill it in with fancy numpy indicing :-)
    dW[np.arange(N), :, y] = dW1
    # Now we average, since we averaged in the loss
    dW = np.mean(dW, axis=0)

    # ----------------------------------------
    # For the b (N, C)
    #
    # As above, we simply need to look at loss per class and put 1s
    db = np.reshape(loss_c > 0, (N, C)).astype(float)
    # Now we fill the k=y part
    db[np.arange(N), y] = -K
    # Now we average, since we averaged in the loss
    db = np.mean(db, axis=0)

    return dW, db

#
# linear_svm.py ends here
