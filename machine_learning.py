import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


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
    (The link no longer works.)
    """
    return tf.keras.Sequential([
        layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, [3, 3], activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])


def create_classical_model2(input_shape: tp.Tuple[int, int], num_classes: int = 1) -> tf.keras.Sequential:
    """A newer classical network
    Based on
    https://keras.io/examples/vision/mnist_convnet/
    """
    return tf.keras.Sequential([
        tf.keras.Input(shape=(*input_shape, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # This additional layer has been added to fix very slow learning
        tf.keras.layers.Dense(128, activation='softmax'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])


def create_fair_classical_model(input_size: tp.Tuple[int, int]) -> tf.keras.Sequential:
    """A simple model based off LeNet
    From https://keras.io/examples/mnist_cnn/
    (The link no longer works.)
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(*input_size, 1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model
