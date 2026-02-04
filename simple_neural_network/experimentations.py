"""
Model Architecture Experimentation Summary
==========================================

Overview of Changes:
--------------------
| Parameter              | Original | Current |
|------------------------|----------|---------|
| HIDDEN_LAYER_1_UNITS   | 128      | 32      |
| HIDDEN_LAYER_2_UNITS   | 64       | 64      |
| Activation (hidden)    | relu     | tanh    |

Current Architecture: 3072 -> 32 (tanh) -> 64 (tanh) -> 1 (sigmoid)


Experiment 1: Hidden Layer 1 Size (128 → 32)
--------------------------------------------

Parameter Impact:
| Layer          | 128 units  | 32 units   |
|----------------|------------|------------|
| Hidden Layer 1 | 393,344    | 98,336     |
| Hidden Layer 2 | 8,256      | 2,112      |
| Total          | ~401K      | ~100K      |
| Model size     | ~4.6 MB    | ~1.2 MB    |

Trade-offs:
    Smaller (32): Faster training, less overfitting, may underfit
    Larger (128): More capacity, slower, risk of overfitting


Experiment 2: Hidden Layer 2 Size (unchanged at 64)
---------------------------------------------------

Current setup (32 -> 64 -> 1):
    - "Expanding" architecture (unusual)
    - Layer 2 has MORE neurons than layer 1

Recommended alternatives:
    - 32 -> 16 -> 1 (true funnel, minimal)
    - 64 -> 32 -> 1 (balanced funnel)


Experiment 3: Activation Function (ReLU → Tanh)
-----------------------------------------------

Comparison:
| Aspect             | ReLU          | Tanh         |
|--------------------|---------------|--------------|
| Output range       | [0, ∞)        | [-1, 1]      |
| Vanishing gradient | Less prone    | More prone   |
| Zero-centered      | No            | Yes          |
| Computation        | Faster        | Slower       |
| Modern usage       | Preferred     | Legacy       |

Considerations:
    - ReLU is standard for hidden layers (post-2012)
    - Tanh may cause slower convergence
    - Tanh useful when zero-centered outputs needed
    - For shallow networks (2 layers), difference is minimal


Recommendations:
----------------
1. Consider funnel architecture: reduce layer 2 to 16 units
2. Test ReLU vs Tanh and compare validation accuracy
3. Monitor for overfitting (train acc >> val acc)
4. For better image classification, consider CNN architecture


Next Experiments to Try:
------------------------
1. Architecture: 32 -> 16 -> 1 (funnel shape)
2. Architecture: 64 -> 32 -> 1 (larger funnel)
3. Activation: Compare relu vs tanh vs leaky_relu
4. Regularization: Add dropout between layers
"""
