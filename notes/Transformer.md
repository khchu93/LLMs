# Transformer

## Motivation
Previously, RNNs, LSTMs, and GRUs were the state-of-the-art approaches for sequence modeling and transduction problems such as machine translation. However, they are slow to train and during inference because of their sequential nature, which prevents parallelization. There were attempts to solve this problem with CNNs, but they had drawbacks related to the loss of accuracy when learning long-range dependencies.
Therefore, the authors proposed a new architecture that dispenses with recurrence and convolutions entirely, relying solely on an attention mechanism. This architecture is capable of:
1. Allowing for significant parallelization = faster training times
2. Reducing the path length for learning long-range dependencies to a constant number of operations = easier to make connections between distant words

## Transformer Architecture Overview
<img src="https://github.com/khchu93/NoteImage/blob/main/Transformer.PNG" alt="transformer" width="600"/>

The **Transformer** is a neural network architecture introduced in *“Attention is All You Need” (Vaswani et al., 2017)*.  
It is the foundation of modern Large Language Models (LLMs) like GPT and BERT.

Transformers are composed of two main modules:

- **Encoder** – reads and understands the input sequence  
- **Decoder** – generates an output sequence (used in translation, summarization, etc.)

Both the encoder and decoder are built from a **stack of N = 6 identical layers** (in the original paper).

### 1. Preparing the Input

Before feeding data into the model, raw text must be converted into numbers through three preprocessing steps.

#### 1.1 Tokenization
Tokenization breaks text into smaller units called **tokens**, which the model can understand.

**Common Tokenization Methods:**

| Type | Example | Pros | Cons |
|------|---------|------|------|
| Word-level | "cats" | Simple | Huge vocabulary, cannot handle new words |
| Character-level | "c", "a", "t", "s" | Small vocab, no unknown words | Sequences too long |
| Subword-level | "cat", "s" | Handles unknown words, compact vocab | More complex |

**Subword Algorithms:**

- **BPE (Byte Pair Encoding)**  
  - Start with characters → merge most frequent pairs → build larger subwords.  
  - Greedy: merges the most frequent pairs to minimize token count.  
  - For new words: splits into longest known subwords.

- **Unigram Model**  
  - Start with a large vocabulary → assign probabilities to tokens → remove lowest probability tokens until desired size.  
  - For new words: chooses the segmentation with the highest probability.


#### 1.2 Embedding
Each token is mapped to a **dense vector** that captures its meaning in continuous space:

```text
token_id → embedding_vector
```
These embeddings are learned during training.

#### 1.3 Positional Encoding

Transformers have **no built-in sense of order**, so positional encodings are added to token embeddings to give information about word positions.

- Can be **sinusoidal** (fixed) or **learned** (trainable).

### 2. Encoder

The **encoder** converts token embeddings into **context-aware vectors** that understand relationships between all words in the input.

Each of the N encoder layers has two main sub-layers:

#### 2.1 Multi-Head Self-Attention

- Each token looks at all other tokens to decide which ones to focus on.
- Represented by three vectors:
  - **Q (Query)** – what this token is looking for
  - **K (Key)** – what each token offers
  - **V (Value)** – the actual information content

**Steps**:
1. Compute attention scores: Q · K^T
2. Scale by √(dimension of K)
3. Apply softmax → normalized attention weights
4. Multiply by V → weighted sum, giving context-aware token representation
**Multi-head attention**: runs multiple attention mechanisms in parallel to capture different types of relationships. Outputs are concatenated and linearly combined.

#### 2.2 Feed-Forward Network (FFN)

A two-layer fully connected network applied to each position separately:
```
FC → ReLU → FC
```

#### 2.3 Residual Connections + Layer Normalization
After each sub-layer (attention or FFN), the input is added back to the output (residual connection) and normalized (LayerNorm) to stabilize training.

### 3. Decoder

The decoder takes the encoder’s output and generates the target sequence **one token at a time** (auto-regressive).

Each decoder layer has three sub-layers:

#### 3.1 Masked Multi-Head Self-Attention
- Same as encoder attention but with a **look-ahead mask** to prevent each position from attending to future tokens.

#### 3.2 Encoder–Decoder (Cross) Attention
- Decoder queries encoder output to align input and output sequences:
  - Q = decoder output
  - K, V = encoder output
#### 3.3 Feed-Forward Network + Residual + LayerNorm
- Same as encoder FFN, applied with residual connections and normalization.

### 4. Output Generation (Autoregression)
- Transformers generate output **one token at a time**.
- The next token depends on all previous tokens:

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

<sup>[4]</sup> **Corpus**: a large collection of **text** used for training or analysis<br>
<sup>[5]</sup> **Vocabulary (Vocab)**: the set of unique **tokens** (words, subwords, or characters) that a model knows how to handle
- Corpus -> Tokenization = Vocab, each vocab has a unique int **ID** and an **embedding vector**

## Key Achievements
- Introduce pure attention mechanisms and replace recurrence and convolution-based models by enabling parallel computation that dramatically speeds up training
- Introduce self-attention and the multi-head attention mechanism
- Introduce positional encoding for order awareness without recurrence

## Pros & Cons

Pros
- Parallel processing
- Long-range dependencies
- Flexible length of input/output
- Rich contextual representations 

Cons
- Memory and computation usage depend on the length of inputs (self-attention = O(n<sup>2</sup>), n = sequence length)
- Require massive datasets and GPU resources to reach SOTA accuracy
- easy to overfit without enough data
- Difficult to understand how exactly it works

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
[How Transformers Work: A Detailed Exploration of Transformer Architecture](https://www.datacamp.com/tutorial/how-transformers-work)
