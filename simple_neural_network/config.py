"""
Configuration constants for Dog vs Cat classifier.
"""

# Random seeds for reproducibility
RANDOM_SEED = 42

# CIFAR-10 class indices
CAT_CLASS = 3
DOG_CLASS = 5

# Training parameters
EPOCHS = 20
BATCH_SIZE = 64

# Model architecture
INPUT_SHAPE = 3072  # 32 * 32 * 3
HIDDEN_LAYER_1_UNITS = 32
HIDDEN_LAYER_2_UNITS = 16

# Paths
SAVED_MODELS_DIR = 'saved_models'
MODEL_PATH_H5 = f'{SAVED_MODELS_DIR}/dog_cat_classifier_full.h5'
WEIGHTS_PATH = f'{SAVED_MODELS_DIR}/dog_cat_classifier.weights.h5'
MODEL_PATH_KERAS = f'{SAVED_MODELS_DIR}/dog_cat_classifier.keras'
