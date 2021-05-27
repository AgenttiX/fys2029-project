import typing as tp

import numpy as np
import tensorflow as tf


def hinge_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Accuracy evaluation based on the hinge loss function
    https://en.wikipedia.org/wiki/Hinge_loss

    :param y_true: true labels
    :param y_pred: predicted labels
    :return:
    """
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


def create_classical_model() -> tf.keras.Sequential:
    """A simple model based off LeNet
    From https://keras.io/examples/mnist_cnn/
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))
    return model


def create_fair_classical_model(input_size: tp.Tuple[int, int]) -> tf.keras.Sequential:
    """A simple model based off LeNet
    From https://keras.io/examples/mnist_cnn/
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(*input_size, 1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model
