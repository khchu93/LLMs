# Transformer

## Motivation
Previously, RNNs, LSTMs, and GRUs were the state-of-the-art approaches for sequence modeling and transduction problems such as machine translation. However, they are slow to train and during inference because of their sequential nature, which prevents parallelization. There were attempts to solve this problem with CNNs, but they had drawbacks related to the loss of accuracy when learning long-range dependencies.
Therefore, the authors proposed a new architecture that dispenses with recurrence and convolutions entirely, relying solely on an attention mechanism. This architecture is capable of:
1. Allowing for significant parallelization = faster training times
2. Reducing the path length for learning long-range dependencies to a constant number of operations = easier to make connections between distant words

## Architecture
<img src="https://github.com/khchu93/NoteImage/blob/main/Transformer.PNG" alt="transformer" width="600"/>

The transformer can be divided into two modules: Encoder and Decoder. Both modules are composed of a stack of N = 6 identical layers. <br>
 
Before the input is fed into the encoder, it has to go through three steps:
1. **Tokenization**: convert raw text into smaller units, **tokens**, which the model can understand and manipulate. The 3 most common types of tokenization are:
- Word level (e.g., "cats") -> simple, but huge vocab size, can't handle new words, fails on unknown words
- Character Level (e.g., "c", "a", "t", "s") -> small vocab, no unknown vocab, long sequences
- Subword (e.g., "cat", "s") -> in between word and character tokenization, can handle unknown words by splitting into known subwords, two ways to do:
- - BPE(Byte Pair Encoding), greedily matching the longest subwords (ensure minimal number of tokens)
  - 1. Split a corpus of text into **characters** (e.g., "low" -> ["l","o","w"])
    2. Count all **adjacent symbol pairs** in the corpus (e.g., ("l","o")=2, ("o","w")=2, ("w","e")=1)
    3. Merge the **most frequent pair** into a new symbol (e.g., "l" + "o" → "lo", "o" + "w" → "ow")
    4. Repeat the process until the desired size of vocabulary is reached, and the vocabulary is ready
  - - When it encounters new words, it will pick the longest subword from the vocab and break it. Repeat until all are known.
  - Unigram
  - 1. Start with a large vocabulary of subwords
    2. Assign a likelihood/probability to each token
    3. Remove the tokens with the lowest likelihood until a desired size of vocabulary is reached
  - - When it encounters new words, it tries different segmentations and chooses the sequence of subwords with the highest probability
2. **Embedding**: convert **tokens** to a vector of numbers that capture the semantic meaning of the token
3. **Positional embedding**: inject positional information into the embeddings of each vector

**Encoder**: maps an input sequence of symbols to a continuous representation/vector space.
- Each of the N encoder layers has two sub-layers:
- 1. Multi-head self-attention mechanism<sup>[1]</sup>
- - 1. dot product matrix multiplication (MatMul) between Query and Key = scores (degree of emphasis each word should place on other words)
    2. reduce the magnitude of scores by the square root of the dimension of query and key vectors, to ensure stable gradients
    3. apply softmax to adjust the score, to ensure all attention weights sum to 1, and highlight important tokens while suppressing irrelevant ones
    4. multiply the softmax output by value, only tokens with a high value are preserved
    - Q = what this token is looking for, K = what each token offers, V = actual information to retrieve if a token is relevant
  2. Residual connection + layer normalization 
  3. Feed-forward network (FC + ReLU + FC)
  4. Residual connection + layer normalization

**Decoder**: uses the continuous representation to generate an output sequence one symbol at a time in an **auto-regressive**<sup>[3]</sup> manner.
- Each of the N decoder layers has three sub-layers:
- 1. Masked multi-head self-attention: multi-head self-attention + look-ahead mask that masks the future tokens to -inf, ensures that the predictions for a particular position can only depend on known outputs at positions before it
  2. Encoder-Decoder Multi-Head Attention or Cross Attention: apply self-attention with the encoder as Query and Key and the decoder as Value 
  3. Feed-forward network (FC + ReLU + FC)
  4. Residual connection + layer normalization

<sup>[1]</sup> **Multi-head self-attention**: run **self-attention**<sup>[2]</sup> multiple times in parallel, each with different learned Q/K/V projections to learn different types of relationships (e.g., grammar, semantic, positional, etc), all heads are concatenated and combined in the end into a final vector.

<sup>[2]</sup> **Self-Attention**: a mechanism that allows each town in a sequence to determine its context-aware representation by weighting its relationships with other tokens

<sup>[3]</sup> **Autoregression**: a modeling approach that **predicts the next value in a sequence based on its previous values**. This can be expressed mathematically using the **chain rule**, where the probability of the whole sequence is the product of the probabilities of each token conditioned on all preceding tokens.:

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
[How Transformers Work: A Detailed Exploration of Transformer Architecture](https://www.datacamp.com/tutorial/how-transformers-work)
