import collections
import typing as tp

import numpy as np
import tensorflow as tf


def filter_binary(x: np.ndarray, y: np.ndarray, class1: int, class2: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Filter a dataset to include only two classes, and create a boolean y vector as their labels."""
    keep = (y == class1) | (y == class2)
    x, y = x[keep], y[keep]
    y = y == class1
    return x, y


def filter_36(x: np.ndarray, y: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Filter only numbers 3 and 6 from the dataset."""
    return filter_binary(x, y, 3, 6)


def filter_49(x: np.ndarray, y: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    return filter_binary(x, y, 4, 9)


def load_data() -> tp.Tuple[tp.Tuple[np.ndarray, np.ndarray], tp.Tuple[np.ndarray, np.ndarray]]:
    """Load the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    return (x_train, y_train), (x_test, y_test)


def remove_contradicting(xs: np.ndarray, ys: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Filter the given dataset so, that images which correspond to both labels are thrown out."""
    # Defaultdict has a default value for new elements
    # https://docs.python.org/3/library/collections.html#collections.defaultdict
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x, y in zip(xs, ys):
        orig_x[tuple(x.flatten())] = x
        mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
        x = orig_x[flatten_x]
        labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
        else:
            # Throw out images that match more than one label.
            pass

    num_uniq_true = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_false = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of unique images with label True:", num_uniq_true)
    print("Number of unique images with label False:", num_uniq_false)
    print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
    print()
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))

    return np.array(new_x), np.array(new_y)


if __name__ == "__main__":
    load_data()
