import numpy as np
from skimage.color import rgb2hsv
from skimage.feature import hog


def extract_h_histogram(data):
    """Extract Hue Histograms from data.

    Parameters
    ----------
    data : ndarray
        NHWC formatted input images.

    Returns
    -------
    h_hist : ndarray (float)
        Hue histgram per image, extracted and reshaped to be NxD, where D is
        the number of bins of each histogram.

    """

    # TODO: Convert data into hsv's, and take only the h

    # TODO: Create bins to be used

    # TODO: Create histogram

    return h_hist


def extract_hog(data):
    """Extract HOG from data.

    Parameters
    ----------
    data : ndarray
        NHWC formatted input images.

    Returns
    -------
    hog_feat : ndarray (float)
        HOG features per image, extracted and reshaped to be NxD, where D is
        the dimension of HOG features.

    """

    # Using HOG
    # HOG -- without the visualization flag
    print("Extracting HOG...")
    hog_feat = np.asarray([
        hog(_x.mean(axis=-1)) for _x in data
    ])
    # Check that the above process is not wrong
    # plt.figure()
    # _hog_tmp, _hog_img = hog(data[0].mean(axis=-1), visualise=True)
    # plt.imshow(_hog_img, cmap=plt.cm.gray)
    # plt.show()

    return hog_feat.astype(float).reshape(len(data), -1)

