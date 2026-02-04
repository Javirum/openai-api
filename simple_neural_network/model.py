"""
Model architecture for Dog vs Cat classifier.
"""

from tensorflow import keras
from tensorflow.keras import layers

from config import INPUT_SHAPE, HIDDEN_LAYER_1_UNITS, HIDDEN_LAYER_2_UNITS


def create_model():
    """
    Build and compile the dog vs cat classifier.

    Returns:
        keras.Model: Compiled model ready for training
    """
    model = keras.Sequential([
        layers.Dense(
            HIDDEN_LAYER_1_UNITS,
            activation='relu',
            input_shape=(INPUT_SHAPE,),
            name='hidden_layer_1'
        ),
        layers.Dense(
            HIDDEN_LAYER_2_UNITS,
            activation='relu',
            name='hidden_layer_2'
        ),
        layers.Dense(
            1,
            activation='sigmoid',
            name='output_layer'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def print_model_info(model):
    """Print model architecture summary."""
    print("\nModel Architecture:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
