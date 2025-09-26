# Transformer

## Motivation
Previously, RNNs, LSTMs, and GRUs were the state-of-the-art approaches for sequence modeling and transduction problems such as machine translation. However, they are slow to train and during inference because of their sequential nature, which prevents parallelization. There were attempts to solve this problem with CNNs, but they had drawbacks related to the loss of accuracy when learning long-range dependencies.
Therefore, the authors proposed a new architecture that dispenses with recurrence and convolutions entirely, relying solely on an attention mechanism. This architecture is capable of:
1. Allowing for significant parallelization = faster training times
2. Reducing the path length for learning long-range dependencies to a constant number of operations = easier to make connections between distant words

## Architecture
<img src="https://github.com/khchu93/NoteImage/blob/main/Transformer.PNG" alt="transformer" width="600"/>

The transformer can be divided into two modules: Encoder and Decoder. Both modules are composed of a stack of N = 6 identical layers. <br>

**Encoder**: maps an input sequence of symbols to a continuous representation/vector space.
- Each of the N encoder layers has two sub-layers:
- 1. Multi-head self-attention mechanism
  2. Position-wise fully connected feed-forward network

**Decoder**: uses the continuous representation to generate an output sequence one symbol at a time in an **auto-regressive**<sup>[1]</sup> manner.
- Each of the N decoder layers has three sub-layers:
- 1. Masked multi-head self-attention
  2. Multi-head attention
  3. Position-wise fully connected feed-forward network

**Multi-head self-attention**
**Self-Attention**
 
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

<sup>[1]</sup> **Autoregression**: a modeling approach that **predicts the next value in a sequence based on its previous values**. This can be expressed mathematically using the **chain rule**, where the probability of the whole sequence is the product of the probabilities of each token conditioned on all preceding tokens.:

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

<sup>[2]</sup> **Corpus**: a large collection of **text** used for training or analysis<br>
<sup>[3]</sup> **Vocabulary (Vocab)**: the set of unique **tokens** (words, subwords, or characters) that a model knows how to handle
- Corpus -> Tokenization = Vocab, each vocab has a unique int **ID** and an **embedding vector**

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
