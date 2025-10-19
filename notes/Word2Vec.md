# Word2Vec

## Motivation

The motivation of the paper centers on **overcoming the computational and data limitations** of existing methods for **learning distributed representations of words while maximizing accuracy, especially for capturing subtle syntactic and semantic regularities**. Many traditional NLP systems treat words as atomic units, such as in the popular `N-gram` model, which **ignores the notion of similarity between words**. While these simple models benefit from robustness and training on massive amounts of data, their **potential for progress is limited** when facing tasks constrained by the size of relevant in-domain data, such as in speech recognition or machine translation. More complex models, particularly those based on neural networks (like `Feedforward Neural Network Language Models (NNLM)` and `Recurrent Neural Network Language Models` (RNNLM)), use **distributed representations of words** and significantly **outperform N-gram models**. However, these previous complex architectures are **computationally expensive**. Specifically, they had **not been successfully** trained on datasets exceeding a **few hundred million words using typical low vector dimensionalities** (50–100), and their complexity is dominated by the non-linear hidden layer, creating a **bottleneck for scaling up to billions of words**.

To address these limitations, the authors' main goal is to **introduce techniques capable of learning high-quality word vectors efficiently from huge data sets**—in the order of billions of words—**and large vocabularies**. The authors propose **two novel model architectures based on log-linear models** that aim to **minimize computational complexity by removing the expensive non-linear hidden layer** found in traditional `NNLMs`. These proposed models are the `Continuous Bag-of-Words (CBOW)` model and the `Continuous Skip-gram` model. 
- The `CBOW` model **predicts the current word based on the continuous distributed representation of the surrounding context**, effectively sharing the projection layer and averaging word vectors.
- The `Skip-gram` model, conversely, **uses the current word as an input to predict surrounding words**.

These simpler architectures allow for **massive improvements in accuracy at a much lower computational cost**, enabling the training of high-quality, high-dimensional word vectors from datasets containing billions of words in less than a day. Furthermore, the architectures are **designed to maximize the accuracy of vector operations, ensuring that the resulting vectors preserve linear regularities among words**, which is **key for measuring and capturing both syntactic and semantic similarities** (e.g., analogical reasoning like "King" - "Man" + "Woman" = "Queen").

## Architecture
![alt text](https://github.com/khchu93/NoteImage/blob/main/word2vec.PNG) <br>
The paper introduces **two novel log-linear model architectures** designed to **efficiently learn distributed representations of words**, specifically by **minimizing computational complexity compared to previous neural network models**. The core innovation of these **"New Log-linear Models"** is the **removal of the expensive non-linear hidden layer** present in traditional `neural network language` models (NNLM). These new architectures, derived from earlier work, **focus exclusively on the step where continuous word vectors are learned using a simple model**.

The two proposed architectures are:

1. `Continuous Bag-of-Words` (CBOW) Model:
   - **Architecture**: The CBOW model is **similar** to a `feedforward NNLM` but **lacks the non-linear hidden layer**.
   - **Functionality**: The **training criterion** for CBOW is to **correctly classify the current (middle) word based on the surrounding context words**.
   - **Input and Projection**: The projection layer is shared for all input words (context words), meaning **all context words are projected into the same position, and their vectors are averaged**. Because the **order of words in the history (context) does not influence** the projection, it is referred to as a `"bag-of-words"` model, though it uses continuous distributed representations. The authors found the best performance using four future words and four history words as input.
   - **Output**: The output layer typically **computes the probability distribution over the entire vocabulary**, often using a **hierarchical softmax** where the vocabulary is represented as a `Huffman binary tree` to reduce computational load.
   - **Computational Complexity** (Q): The complexity per training example is $Q=N×D+D×log_2(V)$, where N is the context size, D is the vector dimensionality, and V is the vocabulary size.

2. `Continuous Skip-gram` Model:
   - **Architecture**: This model is **similar** to the 'CBOW' model in structure but **reverses the prediction task**.
   - **Functionality**: Instead of predicting the current word from the context, the `Skip-gram` model **uses the current word as input to predict the surrounding words (the context) within a specified range**. The goal is to **maximize the classification of a context word based on the input word**.
   - **Input and Prediction**: For a given current word, the model **predicts words that fall within a certain maximum distance (C) before and after the current word**. **Increasing the range** (C) **improves the quality** of the resulting word vectors but also **increases computational complexity**. To mitigate this, **more distant words are given less weight** by sampling less from them in training examples.
   - **Computational Complexity** (Q): The complexity per training example is proportional to $Q=C×(D+D×log_2(V))$, where C is the maximum distance of the words, D is the vector dimensionality, and V is the vocabulary size.
       - If C=10 is chosen, for each training word, a random number R between 1 and C is selected, and 2R word classifications are performed (using the current word as input and R history words and R future words as outputs).

## Key Achievements
- Introduced two novel log-linear model architectures, `CBOW` and `Skip-gram`, by removing the computationally expensive non-linear hidden layer found in traditional Neural Network Language Models (NNLMs) and Recurrent Neural Network Language Models (RNNLMs). This simplification drastically minimized computational complexity.
- The resulting word vectors demonstrated an ability to capture subtle syntactic and semantic regularities among words, measured using simple algebraic operations on the vectors (e.g., vector(”King”) − vector(”Man”) + vector(”Woman”) = vector(”Queen”)).
- 
## Pros & Cons

Pros
- achieved high accuracy at a much lower computational cost by removing the expensive non-linear hidden layer that was the computational bottleneck in traditional Feedforward Neural Network Language Models (NNLM)
- achieved comparable or better semantic accuracy while training over 100× faster, learning from 1.6 billion words in under a day compared to eight weeks for 230 million words with RNNLM.

Cons
- removed the non-linear hidden layer, resulting in a simpler log-linear model that might not be able to represent the data as precisely as complex neural networks.
- Performance of the Skip-gram model is exceptional on semantic tasks, but slightly worse on the syntactic task compared to the CBOW model.
- Performance of the CBOW model is about the same as the NNLM on the semantic task, but is better on the syntactic tasks.

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


