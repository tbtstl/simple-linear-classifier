import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--data_dir", type=str,
                       default="/Users/kwang/Downloads/cifar-10-batches-py",
                       help="Directory with CIFAR10 data")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=100,
                       help="Size of each training batch")

train_arg.add_argument("--num_epoch", type=str,
                       default=100,
                       help="Number of epochs to train")

train_arg.add_argument("--cross_validate", type=str2bool,
                       default=False,
                       help="Whether to perform the cross validation loop")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--feature_type", type=str,
                       default="hog",
                       help="Type of feature to be used")

model_arg.add_argument("--model_type", type=str,
                       default="linear_svm",
                       help="Type of classifier model to use")

model_arg.add_argument("--reg_lambda", type=float,
                       default=1.0,
                       help="Regularization strength")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

