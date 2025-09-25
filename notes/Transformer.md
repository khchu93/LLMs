# Transformer

## Motivation
Previously, RNNs, LSTMs, and GRUs were the state-of-the-art approaches for sequence modeling and transduction problems such as machine translation. However, they are slow to train and during inference because of their sequential nature, which prevents parallelization. There were attempts to solve this problem with CNNs, but they had drawbacks related to the loss of accuracy when learning long-range dependencies.
Therefore, the authors proposed a new architecture that dispenses with recurrence and convolutions entirely, relying solely on an attention mechanism. This architecture is capable of:
1. Allowing for significant parallelization = faster training times
2. Reducing the path length for learning long-range dependencies to a constant number of operations = easier to make connections between distant words

## Architecture
<img src="https://github.com/khchu93/NoteImage/blob/main/Transformer.PNG" alt="transformer" width="600"/>


## Key Achievements
- 

## Pros & Cons

Pros
- 

Cons
- 

## When to use

## Implementation
- Framework: 
- Dataset: 
- Colab Notebook: [link]()

<!--
## Results
Training

Validation

Examples:
-->

## References
