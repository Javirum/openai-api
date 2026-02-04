"""
Evaluation and prediction functions for Dog vs Cat classifier.
"""

import numpy as np


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model performance on test set.

    Args:
        model: Trained Keras model
        x_test: Test features
        y_test: Test labels

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    return test_loss, test_accuracy


def predict(model, x):
    """
    Make predictions on input data.

    Args:
        model: Trained Keras model
        x: Input features

    Returns:
        tuple: (predictions, predicted_classes)
    """
    predictions = model.predict(x, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    return predictions, predicted_classes


def predict_single(model, image):
    """
    Predict class for a single image.

    Args:
        model: Trained Keras model
        image: Single image (flattened or not)

    Returns:
        tuple: (predicted_class, confidence)
               predicted_class: 'Cat' or 'Dog'
               confidence: probability of predicted class
    """
    img = image.reshape(1, -1)
    pred = model.predict(img, verbose=0)[0][0]
    predicted_class = 'Dog' if pred > 0.5 else 'Cat'
    confidence = pred if pred > 0.5 else 1 - pred
    return predicted_class, confidence


def verify_accuracy(predicted_classes, y_test):
    """Manually verify prediction accuracy."""
    correct = (predicted_classes == y_test.flatten()).sum()
    total = len(y_test)
    print(f"Manual accuracy check: {correct}/{total} ({correct/total*100:.2f}%)")
    return correct / total
