"""
Visualization functions for Dog vs Cat classifier.
"""

import matplotlib.pyplot as plt


def plot_sample_images(images, labels, save_path=None):
    """
    Plot a grid of sample images.

    Args:
        images: Array of images
        labels: Corresponding labels (0=Cat, 1=Dog)
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i in range(10):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(images[i])
        label = 'Cat' if labels[i] == 0 else 'Dog'
        axes[row, col].set_title(label)
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample images to '{save_path}'")

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras training history object
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to '{save_path}'")

    plt.show()


def plot_predictions(model, images, labels, save_path=None):
    """
    Plot predictions on sample images.

    Args:
        model: Trained Keras model
        images: Test images (original shape, not flattened)
        labels: True labels
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i in range(10):
        row = i // 5
        col = i % 5

        # Get prediction
        img = images[i].reshape(1, -1)
        pred = model.predict(img, verbose=0)[0][0]
        pred_class = 'Dog' if pred > 0.5 else 'Cat'
        confidence = pred if pred > 0.5 else 1 - pred

        # Actual label
        actual = 'Dog' if labels[i] == 1 else 'Cat'
        correct = '✓' if (pred > 0.5) == (labels[i] == 1) else '✗'

        axes[row, col].imshow(images[i])
        axes[row, col].set_title(f'{pred_class} ({confidence:.2f})\nActual: {actual} {correct}')
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions to '{save_path}'")

    plt.show()
