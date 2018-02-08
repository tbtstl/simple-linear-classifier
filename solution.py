import matplotlib.pyplot as plt
import numpy as np

from config import get_config, print_usage
from utils.cifar10 import load_data
from utils.preprocess import normalize
from utils.regularizor import l2_grad, l2_loss


def _shuffle(a, b):
    assert len(a) == len(b), "Arrays must be of equal length."
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def compute_loss(W, b, x, y, config):
    """Computes the losses for each module."""

    # Lazy import of propper model
    if config.model_type == "linear_svm":
        from utils.linear_svm import model_loss
    elif config.model_type == "logistic_regression":
        from utils.logistic_regression import model_loss
    else:
        raise ValueError("Wrong model type {}".format(
            config.model_type))

    loss, loss_c, pred = model_loss(W, b, x, y)
    loss += config.reg_lambda * l2_loss(W)

    return loss, loss_c, pred


def compute_grad(W, x, y, loss_c, config):
    """Computes the gradient for each module."""

    # Lazy import of propper model
    if config.model_type == "linear_svm":
        from utils.linear_svm import model_grad
    elif config.model_type == "logistic_regression":
        from utils.logistic_regression import model_grad
    else:
        raise ValueError("Wrong model type {}".format(
            config.model_type))

    dW, db = model_grad(loss_c, x, y)
    dW += config.reg_lambda * l2_grad(W)

    return dW, db


def predict(W, b, x, config):
    """Predict function.

    Lazy imports the proper `model_parse` function and returns its
    results. behaves quite similarly to how `compute_loss` and `compute_grad`
    works.

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

    config : namespace
        Arguments and configurations parsed by `argparse`

    Returns
    -------
    pred : ndarray
        Predictions from the model. N numbers where each number corresponds to
        a class.

    """

    # Lazy import of proper model
    model = None

    if config.model_type == 'linear_svm':
        import utils.linear_svm as model
    elif config.model_type == 'logistic_regression':
        import utils.logistic_regression as model

    # use model_predict
    pred = model.model_predict(W, b, x)
    return pred

def train(x_tr, y_tr, x_va, y_va, config):
    """Training function.

    Parameters
    ----------
    x_tr : ndarray
        Training data.

    y_tr : ndarray
        Training labels.

    x_va : ndarray
        Validation data.

    y_va : ndarray
        Validation labels.

    config : namespace
        Arguments and configurations parsed by `argparse`

    Returns
    -------
    train_res : dictionary
        Training results stored in a dictionary file. It should contain W and b
        when best validation accuracy was achieved, as well as the average
        losses per epoch during training, and the average accuracy of each
        epoch to analyze how training went.
    """

    # ----------------------------------------
    # Preprocess data

    # Report data statistic
    print("Training data before: mean {}, std {}, min {}, max {}".format(
        x_tr.mean(), x_tr.std(), x_tr.min(), x_tr.max()
    ))

    # Normalize data using the normalize function. Note that we are remembering
    # the mean and the range of training data and applying that to the
    # validation/test data later on.
    x_tr_n, x_tr_mean, x_tr_range = normalize(x_tr)
    x_va_n, _, _ = normalize(x_va, x_tr_mean, x_tr_range)
    # Always a good idea to print some debug messages
    print("Training data after: mean {}, std {}, min {}, max {}".format(
        x_tr_n.mean(), x_tr_n.std(), x_tr_n.min(), x_tr_n.max()
    ))

    # ----------------------------------------
    # Initialize parameters of the classifier
    print("Initializing...")
    num_class = 10

    # Initialize W to very small random values.
    W = np.random.rand(np.prod(x_tr_n.shape[1:]), num_class) - 0.5

    # Initialize b to zeros
    b = np.zeros(num_class)

    print("Testing...")
    # Test on validation data
    prediction = predict(W, b, x_va_n, config)
    # prediction = prediction.reshape(y_va.shape)
    correct_pred_count = np.sum(prediction == y_va)

    acc = (correct_pred_count/float(y_va.shape[0]))*100
    print("Initial Validation Accuracy: {}%".format(acc))

    batch_size = config.batch_size
    num_epoch = config.num_epoch
    num_batch = len(x_tr_n) // batch_size
    loss_epoch = []
    tr_acc_epoch = []
    va_acc_epoch = []
    W_best = None
    b_best = None
    best_acc = 0
    # For each epoch
    for idx_epoch in range(num_epoch):
        # Create a random order to go through the data
        order = np.arange(num_batch)
        np.random.shuffle(order)

        losses = np.zeros(num_batch)
        accs = np.zeros(num_batch)
        for idx_batch in range(num_batch):
            # Construct batch
            idx = order[idx_batch]
            if idx+batch_size > y_tr.shape[0]:
                x_b = np.copy(x_tr_n[idx:])
                y_b = np.copy(y_tr[idx:])
            else:
                x_b = np.copy(x_tr_n[idx:idx+batch_size])
                y_b = np.copy(y_tr[idx:idx+batch_size])

            # Get loss with compute_loss
            loss_cur, loss_c, pred_b = compute_loss(W, b, x_b, y_b, config)
            # Get gradient with compute_grad
            dW, db = compute_grad(W, x_b, y_b, loss_c, config)

            # Update parameters
            W -= (dW*config.learning_rate)
            b -= (db*config.learning_rate)

            # Record this batches result
            correct_pred_count = np.sum(pred_b == y_b)
            acc = (correct_pred_count / float(y_b.shape[0]))

            losses[idx_batch] = loss_cur
            accs[idx_batch] = acc

        # Report average results within this epoch
        print("Epoch {} -- Train Loss: {}".format(
            idx_epoch, np.mean(losses)))
        print("Epoch {} -- Train Accuracy: {:.2f}%".format(
            idx_epoch, np.mean(accs) * 100))

        # Test on validation data and report results
        prediction = predict(W, b, x_va_n, config)
        correct_pred_count = np.sum(prediction == y_va)
        acc = (correct_pred_count / float(y_va.shape[0]))

        print("Epoch {} -- Validation Accuracy: {:.2f}%".format(
            idx_epoch, acc * 100))

        # If best validation accuracy, update W_best, b_best, and best
        # accuracy. We will only return the best W and b
        if acc > best_acc:
            W_best = W
            b_best = b
            best_acc = acc

        # Record per epoch statistics
        loss_epoch += [losses.mean()]
        tr_acc_epoch += [accs.mean()]
        va_acc_epoch += [acc]

    # Pack results. Remeber to pack pre-processing related things here as
    # well
    train_res = {
        'W_best': W_best,
        'b_best': b_best,
        'best_acc': acc,
        'loss_epoch': loss_epoch,
        'tr_acc_epoch': tr_acc_epoch,
        'va_acc_epoch': va_acc_epoch,
        'x_tr_mean': x_tr_mean,
        'x_tr_range': x_tr_range
    }

    return train_res


def main(config):
    """The main function."""

    # ----------------------------------------
    # Load cifar10 train data
    print("Reading training data...")
    data_trva, y_trva = load_data(config.data_dir, "train")

    # ----------------------------------------
    # Load cifar10 test data
    print("Reading test data...")
    data_te, y_te = load_data(config.data_dir, "test")

    # ----------------------------------------
    # Extract features
    print("Extracting Features...")
    if config.feature_type == "hog":
        # HOG features
        from utils.features import extract_hog
        x_trva = extract_hog(data_trva)
        x_te = extract_hog(data_te)
    elif config.feature_type == "h_histogram":
        # Hue Histogram features
        from utils.features import extract_h_histogram
        x_trva = extract_h_histogram(data_trva)
        x_te = extract_h_histogram(data_te)
    elif config.feature_type == "rgb":
        # raw RGB features
        x_trva = data_trva.astype(float).reshape(len(data_trva), -1)
        x_te = data_te.astype(float).reshape(len(data_te), -1)

    # ----------------------------------------
    # Create folds
    num_fold = 5

    # Randomly shuffle data and labels.
    x_trva, y_trva = _shuffle(x_trva, y_trva)

    # Reshape the data into 5x(N/5)xD, so that the first dimension is the fold
    x_trva = np.reshape(x_trva, (num_fold, len(x_trva) // num_fold, -1))
    y_trva = np.reshape(y_trva, (num_fold, len(y_trva) // num_fold))

    # Cross validation setup. If you set cross_validate as False, it will not
    # do all 5 folds, but report results only for the first fold. This is
    # useful when you want to debug.
    if config.cross_validate:
        va_fold_to_test = np.arange(num_fold)
    else:
        va_fold_to_test = np.arange(1)

    # ----------------------------------------
    # Cross validation loop
    train_res = []
    for idx_va_fold in va_fold_to_test:
        # Select train and validation. Notice that `idx_va_fold` will be
        # the fold that you use as validation set for this experiment
        va_idx = [i for i in range(num_fold) if i != idx_va_fold]
        x_tr = np.delete(x_trva, idx_va_fold, 0)
        x_tr = x_tr.reshape(-1, x_tr.shape[-1])
        y_tr = np.delete(y_trva, idx_va_fold, 0)
        y_tr = y_tr.reshape(np.prod(y_tr.shape))
        x_va = np.delete(x_trva, va_idx, 0)
        x_va = x_va.reshape(-1, x_va.shape[-1])
        y_va = np.delete(y_trva, va_idx, 0)
        y_va = y_va.reshape(np.prod(y_va.shape))

        # ----------------------------------------
        # Train
        print("Training for fold {}...".format(idx_va_fold))
        # Run training
        cur_train_res = train(x_tr, y_tr, x_va, y_va, config)

        # Save results
        train_res += [cur_train_res]

    # TODO: Average results to see the average performance for this set of
    # hyper parameters on the validation set. This will be used to see how good
    # the design was. However, this should all be done *after* you are sure
    # your implementation is working. Do check how the training is going on by
    # looking at `loss_epoch` `tr_acc_epoch` and `va_acc_epoch`
    TODO

    # TODO: Find model with best validation accuracy and test it. Remember you
    # don't want to use this result to make **any** decisions. This is purely
    # the number that you show other people for them to evaluate your model's
    # performance.


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
