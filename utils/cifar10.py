import os

import numpy as np

from utils.external import unpickle


def load_data(data_dir, data_type):
    """Function to load data from CIFAR10.

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing the extracted CIFAR10 files.

    data_type : string
        Either "train" or "test", which loads the entire train/test data in
        concatenated form.

    Returns
    -------
    data : ndarray (uint8)
        Data from the CIFAR10 dataset corresponding to the train/test
        split. The datata should be in NHWC format.

    labels : ndarray (int)
        Labels for each data. Integers ranging between 0 and 9.

    """

    if data_type == "train":
        data = []
        label = []
        for _i in range(5):
            file_name = os.path.join(data_dir, "data_batch_{}".format(_i + 1))
            cur_dict = unpickle(file_name)
            data += [
                np.array(cur_dict[b"data"])
            ]
            label += [
                np.array(cur_dict[b"labels"])
            ]
        # Concat them
        data = np.concatenate(data)
        label = np.concatenate(label)

    elif data_type == "test":
        data = []
        label = []
        cur_dict = unpickle(os.path.join(data_dir, "test_batch"))
        data = np.array(cur_dict[b"data"])
        label = np.array(cur_dict[b"labels"])

    else:
        raise ValueError("Wrong data type {}".format(data_type))

    # Turn data into (NxHxWxC) format, so that we can easily process it, where
    # N=number of images, H=height, W=widht, C=channels. Note that this
    # corresponds to Tensorflow format that we will use later.
    data = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))

    return data, label
