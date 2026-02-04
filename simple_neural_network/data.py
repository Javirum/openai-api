"""
Data loading and preprocessing for Dog vs Cat classifier.
"""

import numpy as np
from tensorflow.keras.datasets import cifar10

from config import CAT_CLASS, DOG_CLASS


def load_cifar10_dogs_cats():
    """
    Load CIFAR-10 dataset and filter for dogs and cats only.

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
               Labels are converted to binary: cat=0, dog=1
    """
    (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

    # Filter training set
    train_mask = (y_train_full.flatten() == CAT_CLASS) | (y_train_full.flatten() == DOG_CLASS)
    x_train = x_train_full[train_mask]
    y_train = y_train_full[train_mask]
    y_train = np.where(y_train == CAT_CLASS, 0, 1)

    # Filter test set
    test_mask = (y_test_full.flatten() == CAT_CLASS) | (y_test_full.flatten() == DOG_CLASS)
    x_test = x_test_full[test_mask]
    y_test = y_test_full[test_mask]
    y_test = np.where(y_test == DOG_CLASS, 1, 0)

    return (x_train, y_train), (x_test, y_test)


def preprocess(x_train, x_test):
    """
    Normalize and flatten images for the neural network.

    Args:
        x_train: Training images
        x_test: Test images

    Returns:
        tuple: (x_train_flat, x_test_flat) normalized and flattened
    """
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten images (32x32x3 = 3072)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    return x_train_flat, x_test_flat


def print_dataset_info(x_train, y_train, x_test, y_test):
    """Print dataset statistics."""
    print(f"Training: {x_train.shape[0]} images")
    print(f"Test: {x_test.shape[0]} images")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Training set - Cats: {(y_train == 0).sum()}, Dogs: {(y_train == 1).sum()}")
    print(f"Test set - Cats: {(y_test == 0).sum()}, Dogs: {(y_test == 1).sum()}")
