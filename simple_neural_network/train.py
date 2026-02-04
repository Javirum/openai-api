"""
Training logic for Dog vs Cat classifier.
"""

from config import EPOCHS, BATCH_SIZE


def train_model(model, x_train, y_train, x_val, y_val):
    """
    Train the model on the provided data.

    Args:
        model: Compiled Keras model
        x_train: Training features
        y_train: Training labels
        x_val: Validation features
        y_val: Validation labels

    Returns:
        History: Training history object
    """
    print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        verbose=1
    )

    print("\nTraining complete!")
    return history
