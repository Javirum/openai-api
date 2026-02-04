"""
Model saving and loading utilities.
"""

import os
from tensorflow import keras

from config import (
    SAVED_MODELS_DIR,
    MODEL_PATH_H5,
    WEIGHTS_PATH,
    MODEL_PATH_KERAS
)


def save_model(model):
    """
    Save model in multiple formats.

    Args:
        model: Trained Keras model
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # Save in H5 format
    model.save(MODEL_PATH_H5)
    print(f"Saved full model to: {MODEL_PATH_H5}")
    print(f"  File size: {os.path.getsize(MODEL_PATH_H5) / 1024 / 1024:.2f} MB")

    # Save weights only
    model.save_weights(WEIGHTS_PATH)
    print(f"Saved weights to: {WEIGHTS_PATH}")
    print(f"  File size: {os.path.getsize(WEIGHTS_PATH) / 1024 / 1024:.2f} MB")

    # Save in native Keras format (recommended)
    model.save(MODEL_PATH_KERAS)
    print(f"Saved in Keras format to: {MODEL_PATH_KERAS}")
    print(f"  File size: {os.path.getsize(MODEL_PATH_KERAS) / 1024 / 1024:.2f} MB")


def load_model(path=MODEL_PATH_KERAS):
    """
    Load a saved model.

    Args:
        path: Path to saved model file

    Returns:
        keras.Model: Loaded model
    """
    return keras.models.load_model(path)


def verify_saved_model(original_model, x_test, y_test):
    """
    Verify that saved model produces same results.

    Args:
        original_model: Original trained model
        x_test: Test features
        y_test: Test labels

    Returns:
        bool: True if models match
    """
    print("\nVerifying saved model...")

    loaded_model = load_model(MODEL_PATH_H5)
    _, test_acc_loaded = loaded_model.evaluate(x_test, y_test, verbose=0)
    _, test_acc_original = original_model.evaluate(x_test, y_test, verbose=0)

    print(f"Loaded model accuracy: {test_acc_loaded:.4f}")
    print(f"Original model accuracy: {test_acc_original:.4f}")

    match = abs(test_acc_loaded - test_acc_original) < 0.001
    print(f"Match: {'✓' if match else '✗'}")

    return match
