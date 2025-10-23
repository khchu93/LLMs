# Bidirectional Encoder Representations from Transformers (BERT)

## Motivation

The motivation of the `BERT` paper stems from the **critical limitations** of existing pre-trained language representation models, particularly their **unidirectional nature**, which **restricts their ability to learn deep representations** crucial for complex Natural Language Processing (NLP) tasks. Current pre-training strategies fall into two categories: **feature-based** (like `ELMo`) and **fine-tuning** (like `OpenAI GPT`). The **fine-tuning approach**, while **requiring minimal task-specific parameters**, is significantly **restricted because standard language models are unidirectional** (e.g., left-to-right). This unidirectional limitation forces models like `OpenAI GPT` to **only attend to previous tokens** in the self-attention layers of the Transformer architecture, which the authors argue is **sub-optimal for sentence-level tasks** and particularly **harmful for token-level tasks** like question answering, where **context from both directions is crucial**. While feature-based models like `ELMo` attempt **bidirectionality by concatenating independently trained left-to-right and right-to-left `Language Models (LMs)`**, this approach results in a **shallow concatenation** that is strictly **less powerful** than a truly deep bidirectional model.

To overcome the unidirectionality constraint and unlock the full power of pre-trained representations, the authors propose `BERT`: **Bidirectional Encoder Representations from Transformers**. `BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers`. This bidirectionality is achieved by introducing a new pre-training objective called the **`"masked language model" (MLM)`**, inspired by the `Cloze task`, which **randomly masks some input tokens and trains the model to predict the original words based on the surrounding context**. Additionally, `BERT` is jointly pre-trained on a **`"next sentence prediction" (NSP)`** task to explicitly **learn text-pair relationships**, which is crucial for downstream tasks like `Question Answering (QA)` and `Natural Language Inference (NLI)`. The combined approach enables the pre-trained BERT model to be fine-tuned with just one additional output layer to achieve state-of-the-art results on a wide range of tasks without substantial task-specific architecture modifications, demonstrating the importance of deep bidirectional pre-training.

## Architecture

<img src="https://github.com/khchu93/NoteImage/blob/main/bert.png" alt="embedding" width="600"/><br>
**Core Architectural Components**<br>
The architecture of `BERT`, which stands for **Bidirectional Encoder Representations from Transformers**, is a **multi-layer bidirectional Transformer encoder**.

The structure is **based on the original Transformer encoder** described by Vaswani et al. (2017). The implementation used in BERT is **"almost identical" to the original** architecture.

1. **Transformer Encoder**: The `BERT` model utilizes the **encoder component of the Transformer**, which is **inherently bidirectional**. Critically, unlike the Transformer used in OpenAI GPT (which is constrained to unidirectional, left-to-right self-attention), the BERT Transformer uses **bidirectional self-attention**. This design **allows every token in every layer to jointly condition on both the left and right context**.

2. **Model Sizes**: The architecture is implemented in **two primary sizes**, with the hyperparameters defined by `L (number of layers/Transformer blocks)`, `H (hidden size)`, and `A (number of self-attention heads)`:
   - BERT<sub>BASE</sub>: L=12, H=768, A=12, totaling 110 million parameters. This size was chosen to match OpenAI GPT for comparison.
   - BERT<sub>LARGE</sub>: L=24, H=1024, A=16, totaling 340 million parameters.

3. **Input/Output Representation** (Sequence Encoding): To **handle various downstream tasks**, the input representation is unified to unambiguously **represent both single sentences and pairs of sentences** (e.g., Question and Answer) **in a single token sequence**. The **final input representation** for a given token is **constructed by summing three types of embeddings**:
    - **`Token Embeddings`**: Derived using **`WordPiece embeddings`** with a 30,000 token vocabulary.
    - **`Segment Embeddings`** (Sentence A/B): A **learned embedding** added to every token to **indicate whether it belongs to the first sentence** (Sentence A) **or the second sentence** (Sentence B) when processing sentence pairs.
    - **`Position Embeddings`**: Standard position embeddings are added to **indicate the order of tokens in the sequence**.

      **Special tokens** are mandatory for **input formatting**:<br>
      - **`[CLS]`**: A special **classification token** inserted as the **first token of every input sequence**. Its **final hidden state** is used as the **aggregate sequence representation** for classification tasks.
      - **`[SEP]`**: A special **separator token** used to denote the **end of a sentence or to separate two sentences packed together** (e.g., separating a question from a paragraph).
        <img src="https://github.com/khchu93/NoteImage/blob/main/bert_embedding.PNG" alt="embedding" width="800"/>

**Pre-training Objectives that Enable Bidirectionality**<br>
The **deep bidirectional nature** of the `BERT` architecture is **enabled by two specific unsupervised pre-training tasks**:
1. **`Masked Language Model (MLM)`**: This objective **randomly masks 15% of the input tokens** and trains the model to **predict the original identity** of the masked tokens **based on their surrounding context (left and right)**. This method **avoids the issue of unidirectional conditioning** and allows for the pre-training of a **truly deep bidirectional Transformer**.

    <img src="https://github.com/khchu93/NoteImage/blob/main/bert_mlm.png" alt="embedding" width="600"/>

2. **`Next Sentence Prediction (NSP)`**: To learn **relationships between two sentences**—crucial for downstream tasks like `Question Answering (QA)` and `Natural Language Inference (NLI)`—`BERT` is trained to **predict whether Sentence B is the actual next sentence following Sentence A (IsNext) or a random sentence (NotNext)**. The **final hidden vector** corresponding to the [CLS] token (C) is used for this **binary classification task**.

**Fine-Tuning Structure**<br>
**A key architectural feature of BERT is its unified architecture for pre-training and fine-tuning, with minimal differences between the two stages**. For downstream tasks, **only one additional output layer** is required, and all parameters are fine-tuned end-to-end. The self-attention mechanism **handles both single text and text pair inputs by encoding the concatenated sequence**  (e.g., [CLS] Sentence A [SEP] Sentence B [SEP]), effectively performing bidirectional cross attention between sentences.

## Key Achievements
- Designed `BERT` to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers of the Transformer encoder, which allows the model to gain full context understanding.
- Introduced the `masked language model (MLM)` objective that randomly masks 15% of the input tokens and trains the model to predict the original identity of the masked words based only on their surrounding context. This mechanism forces the model to learn a deep fusion of left and right context without tokens trivially seeing themselves.
- Introduced the `Next Sentence Prediction (NSP)` task during pre-training. This binarized prediction task explicitly forces the model to learn text-pair relationships (IsNext/NotNext).

## Pros & Cons

Pros
- Better long-range dependency: Through self-attention, **every token can directly connect to every other token**, regardless of position. This eliminates the **vanishing gradient** issue common in `RNNs`/`LSTMs`.
- Better representation learning: BERT’s **embeddings are contextualized**<sup>[1]</sup> — the same word gets different vectors depending on its meaning in context, which leads to **stronger semantic representations**.
- Full context understanding: Each token’s embedding is computed by **attending to both left and right context** simultaneously, which allows BERT to **fully understand meaning and relationships** between words in a sentence.

   [1] Contextual embedding ≠ Input embedding<br>
   The input embedding is the element-wise sum of the token, position, and segment embeddings. The contextual embedding, on the other hand, is the vector output of each token from the final hidden layer, capturing the meaning of the word in the context of the entire sentence.

Cons
- High computational cost: Bidirectional attention **doubles computation** compared to unidirectional models. Original BERT took days on 16 TPUs.
- Slow Inference Speed: Inference is not parallelizable in the same way as autoregressive models because all tokens depend on each other.
- Not Suited for Text Generation: Bidirectional models can’t generate text sequentially since they rely on seeing both sides of a token.
- Memory Intensive: The self-attention mechanism has O(n²) complexity in sequence length. Memory usage explodes once the sequence exceeds 512 tokens.

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

