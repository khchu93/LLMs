# Seq2Seq

## Motivation

The paper is motivated by the **significant limitations of existing models in handling variable-length sequences**, despite the general power of `Deep Neural Networks (DNNs)` on many difficult learning tasks such as speech recognition and visual object recognition. `DNNs` are extremely powerful because they can **perform arbitrary parallel computation**. However, a **significant limitation** is that `DNNs` can only be applied to problems **whose inputs and targets can be encoded with vectors of fixed dimensionality**. This constraint makes them unsuitable for many important problems, such as machine translation, speech recognition, and question answering, which are inherently **sequential problems where sequence lengths are not known a priori**. Thus, while `DNNs` are capable of intricate computations and achieve excellent performance when fixed-dimensional inputs are available, they **cannot directly map sequences to sequences where input and output lengths differ with complicated or non-monotonic relationships**.

To address this challenge, the authors present a general, end-to-end approach to sequence learning using the **`Long Short-Term Memory (LSTM)`** architecture, which is **known to learn problems with long-range temporal dependencies**. The proposed method, referred to as **sequence-to-sequence learning**, **uses one multilayered `LSTM` to map the input sequence to a vector of a fixed dimensionality (the encoder), and another deep `LSTM` to decode the target sequence from that vector (the decoder)**. This architecture allows the model to **estimate the conditional probability** of an output sequence given an input sequence, where the **lengths of the sequences may differ**. <br>
A key technical contribution introduced to **simplify the optimization problem** and markedly **improve performance**—especially on long sentences—was the technique of **reversing the order of words in all source sentences (but not the target sentences) during training**. This simple data transformation **introduces many short-term dependencies** between the source and target, making the **optimization process much easier** for `Stochastic Gradient Descent (SGD)`. The success of this LSTM-based approach is demonstrated by achieving a BLEU score of 34.8 on the WMT’14 English to French translation task, outperforming a phrase-based `Statistical Machine Translation (SMT)` baseline of 33.3 on the same dataset.

## Architecture
![alt text](https://github.com/khchu93/NoteImage/blob/main/seq2seq.webp) <br>
[Source](https://www.guru99.com/seq2seq-model.html)

The `sequence-to-sequence` learning model architecture is built using **two separate, deep `Long Short-Term Memory (LSTM)` networks**: one acting as an **Encoder** and the other as a **Decoder**. This design allows the model to **handle sequences of varying lengths, overcoming the fixed dimensionality constraint** of traditional `Deep Neural Networks (DNNs)`.

1. **The Encoder** (Input Sequence Processing)
  - **Function**: The encoder's purpose is to **read the input sequence**, one timestep at a time, and compress all the information **into a single fixed-dimensional vector representation**.
  - **Architecture Details**: The authors used a **multilayered LSTM** (specifically, four layers) for the encoder. The hidden state of the final LSTM layer, after processing the entire input sequence, becomes the fixed-dimensional vector representation (v). The capacity of this representation is high; the LSTMs used 1000 cells at each of the four layers, resulting in 8,000 real numbers used to represent a sentence.
  - **Input Reversal** (Key Technical Contribution): Crucially, the model is trained by reversing the order of words in all source sentences (the input sequence), but not the target sentences. For example, to map sentence $a,b,c$ to $α,β,γ$, the LSTM is trained to map $c,b,a$ to $α,β,γ$. This simple data transformation significantly improved performance because it introduced many short-term dependencies, making it easier for `Stochastic Gradient Descent (SGD)` to establish communication between the input and output sequences and simplifying the optimization problem.

2. **The Decoder** (Output Sequence Generation)
  - **Function**: The decoder is responsible for generating the target output sequence from the fixed-dimensional vector representation (v) created by the encoder.
  - **Architecture Details**: The decoder is another deep LSTM (also four layers in the authors' specific implementation), which is essentially a recurrent neural network language model. Its initial hidden state is set to the fixed-dimensional representation (v) obtained from the encoder.
  - **Decoding Process**: The decoder then computes the conditional probability of the output sequence $y_1,...,y_{T'}$ by iteratively calculating the probability of the next word $p(y_t \mid v, y_1, \ldots, y_{t-1})$. Each probability distribution is represented with a softmax over the entire vocabulary. The decoding process continues until the model outputs a special end-of-sentence symbol (“”).

The overall scheme uses two distinct LSTMs—one for the input and one for the output—which increases the number of model parameters at negligible computational cost and facilitates training on multiple language pairs. The resulting deep, two-part LSTM architecture forms an end-to-end approach to sequence learning.

## Key Achievements
- Proposed a two-part deep LSTM architecture that enables the model to handle inputs and outputs of different lengths.
- Introduced the technique of reversing the order of words in the source sentences during training, which improved performance by introducing short-term dependencies and facilitating the optimization.

## Pros & Cons

Pros
- Allowed models to map sequence of varying lengths to sequence
- Handled long-range dependencies

Cons
- Out-of-Vocabulary (UNK) Limitation, where the model relies on a fixed, limited vocabulary (160,000 words for source, 80,000 for target) and every word outside this set is replaced with a special “UNK” token.
- Approximation in Decoding, where the system relies on an approximate method (simple left-to-right beam search decoder), rather than an exact solution.

<!--
## Implementation
- Framework: 
- Dataset: 
- Colab Notebook: [link]()

## Results
Training

Validation

Examples:
-->

## References
[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215)
