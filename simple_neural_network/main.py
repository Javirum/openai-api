"""
Dog vs Cat Image Classifier
Author: Javier Romero
Description: Build a simple neural network to classify dog and cat images
"""

import numpy as np
import tensorflow as tf

from config import RANDOM_SEED
from data import load_cifar10_dogs_cats, preprocess, print_dataset_info
from model import create_model, print_model_info
from train import train_model
from evaluate import evaluate_model, predict, verify_accuracy
from visualize import plot_sample_images, plot_training_history, plot_predictions
from save_model import save_model, verify_saved_model


def main():
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))

    # Load and preprocess data
    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)

    (x_train, y_train), (x_test, y_test) = load_cifar10_dogs_cats()
    print_dataset_info(x_train, y_train, x_test, y_test)

    # Keep original images for visualization
    x_train_original = x_train.astype('float32') / 255.0
    x_test_original = x_test.astype('float32') / 255.0

    # Preprocess for model
    x_train_flat, x_test_flat = preprocess(x_train, x_test)
    print(f"Flattened shape: {x_train_flat.shape}")

    # Visualize samples
    # plot_sample_images(x_train_original, y_train, 'sample_images.png')

    # Build model
    print("\n" + "=" * 50)
    print("BUILDING MODEL")
    print("=" * 50)

    model = create_model()
    print_model_info(model)

    # Train model
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)

    history = train_model(model, x_train_flat, y_train, x_test_flat, y_test)

    # Evaluate model
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    evaluate_model(model, x_test_flat, y_test)

    predictions, predicted_classes = predict(model, x_test_flat)
    verify_accuracy(predicted_classes, y_test)

    # Visualize results
    # plot_training_history(history, 'training_history.png')
    # plot_predictions(model, x_test_original, y_test, 'sample_predictions.png')

    # Save model
    print("\n" + "=" * 50)
    print("SAVING MODEL")
    print("=" * 50)

    save_model(model)
    verify_saved_model(model, x_test_flat, y_test)

    print("\nDone!")


if __name__ == "__main__":
    main()
