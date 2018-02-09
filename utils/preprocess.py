# preprocess.py ---
#
# Filename: preprocess.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Jan 15 10:10:03 2018 (-0800)
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
from scipy import stats

import numpy as np


def normalize(data, data_mean=None, data_range=None):
    """Normalizes input data.

    Notice that we have two optional input arguments to this function. When
    dealing with validation/test data, we expect these to be given, since we
    should not learn or change *anything* of the trained model. This includes
    the data preprocessing step.

    Parameters
    ----------
    data : ndarray
        Input data that we want to normalize. NxD, where D is the
        dimension of the input data.

    data_mean : ndarray (optional)
        Mean of the data that we should use. 1xD is expected. If not given,
        this will be computed from `data`.

    data_range : ndarray (optional)
        Maximum deviation from the mean. 1xD is expected. If not given, this
        will be computed from `data`.

    Returns
    -------
    data_n : ndarray
        Normalized data. NxD, where D is the dimension of the input data.

    data_mean : ndarray
        Mean. 1xD.

    data_range : ndarray
        Maximum deviation from the mean. 1xD.

    """

    # Make data float, just in case
    data_f = data.astype(float)

    # Compute mean if needed
    if data_mean is None:
        data_mean = data_f.mean(axis=0)

    # Make zero mean
    std = np.sqrt(np.sum((data_f - data_mean)**2) / data_f.shape[0])
    data_n = (data_f - data_mean) / std

    # Compute maximum deviation from zero if needed
    if data_range is None:
        pos_dev = np.abs(np.max(data_n, axis=0))
        neg_dev = np.abs(np.min(data_n, axis=0))
        data_range = np.maximum(pos_dev, neg_dev)

    # Divide with the range
    data_n /= data_range

    return data_n, data_mean, data_range


#
# preprocess.py ends here
