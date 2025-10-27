# Generative Pre-Training (GPT)

## Motivation

**Current Models' Limitations**<br>
The motivation for the paper stems from the **pervasive dependence** of modern `Natural Language Processing (NLP)` deep learning methods **on substantial amounts of manually labeled data**, which severely **restricts their applicability**, especially in domains lacking annotated resources. While previous approaches have successfully leveraged **unsupervised learning**, primarily through the use of **pre-trained word embeddings**, these methods mainly **transfer only word-level information**. 

**Leveraging more than word-level semantics from unlabeled text is challenging** for existing semi-supervised approaches, largely because there is **no consensus on the most effective optimization objectives** for learning text representations useful for transfer (with various methods like language modeling, machine translation, and discourse coherence showing mixed results), **nor is there consensus on the best way to transfer those representations to target tasks** without requiring substantial task-specific architectural modifications. Existing transfer techniques often **involve adding intricate learning schemes, auxiliary objectives, or task-specific changes** to the model architecture.

**Proposed Solution**<br>
To address the scarcity of labeled data and the uncertainties surrounding effective transfer learning, the authors explore a **semi-supervised approach** utilizing a **two-stage training procedure: unsupervised pre-training followed by supervised fine-tuning**. `The central goal is to learn a universal representation that can transfer with minimal adaptation to a wide range of language understanding tasks`. <br>
- The authors propose using a **generative pre-training** of a language model on a **diverse corpus of unlabeled text** to learn the **initial parameters of a neural network**. <br>
- For the model architecture, they employ the **Transformer decoder network**, which **provides a more structured memory for handling long-term dependencies** compared to alternatives like recurrent networks (e.g., LSTMs). <br>
- Crucially, during the discriminative fine-tuning stage, the authors introduce **task-aware input transformations that convert structured text inputs** (like sentence pairs for textual entailment or question/answer triplets) **into a single contiguous token sequence**. This approach **minimizes changes to the model architecture during transfer**, allowing the same general task-agnostic model to significantly improve the state of the art in 9 out of 12 studied tasks.

## Architecture
<img src="https://github.com/khchu93/NoteImage/blob/main/gpt.PNG" alt="embedding" width="600"/><br>

The Generative Pre-Training (GPT) model architecture is based on a **two-stage training procedure** using a **high-capacity Transformer decoder network**. The architecture is designed to **facilitate robust transfer performance across diverse tasks with minimal modification**.

1. **Core Model Architecture** (The Transformer Decoder)<br>
The model utilizes a **multi-layer Transformer decoder**, which is a variant of the original Transformer model. The Transformer was chosen over alternatives like recurrent networks (e.g., LSTMs) because it **provides a more structured memory** by allowing every position in the input sequence to attend to every other position in the sequence simultaneously, **enabling the model to effectively handle long-term dependencies in text**.<br><br>
The specific model architecture used in the experiments is a **12-layer decoder-only transformer**. Key specifications include:
    - **Masked Self-Attention**: The model uses **masked self-attention** heads to maximize the likelihood of predicting the current token ($u_i$) given only the previous k context tokens ($u_{i-k}, \ldots, u_{i-1}$). This operation is applied over the input context tokens to produce an output probability distribution over the vocabulary for each position.<br>
    - **Dimensions**: It uses **768-dimensional states** and **12 attention heads**.
    - **Feed-forward Networks**: The position-wise feed-forward networks use **3072-dimensional inner states**.
    - **Embeddings**: The inputs are processed using a `bytepair encoding (BPE)` **vocabulary with 40,000 merges**. The model uses **learned position embeddings** instead of the sinusoidal version found in the original Transformer work.
    - **Regularization and Activation**: `Dropout` is applied for **regularization**, and the `Gaussian Error Linear Unit (GELU)` is used as the activation function.

The model processes the input context vector of tokens ($U$) through n layers ($h_l$), where $h_0$ is the sum of the token embedding matrix ($W_e$​) and the position embedding matrix ($W_p$).<br>
> $h_0 = UW_e + W_p$<br>
> $h_l = \text{TransformerBlock}(h_{l-1})$<br>
> $P(u) = \text{softmax}(h_n W_e^{T})$<br>

2. **Two-Stage Training Architecture**<br>
The overall framework is a **semi-supervised** approach comprising two distinct stages:
    1. **Unsupervised Pre-training** (Language Modeling)<br>
        - **Objective**: The model is trained on a **large, unlabeled corpus** (like the BooksCorpus) to **maximize the standard language modeling objective** ($L_1$), which is the likelihood of predicting the next token given the previous k context tokens. This stage learns the **initial parameters** ($Θ$) of the neural network.<br>
    2. **Supervised Fine-tuning** (Task Adaptation)<br>
        - **Objective**: The pre-trained parameters are adapted to a **specific labeled** target task by **maximizing a supervised objective** ($L_2$).<br>
        - **Classification Layer**: The final Transformer block’s activation ($h^l_m$)—derived from the sequence of input tokens ($x_1,…,x_m$)—is fed into an **added linear output layer** with parameters $W_y$ to **predict the label** $y$.<br>
        - **Auxiliary Objective** (Optional): The authors found that including the **language modeling objective** ($L_1$) as an **auxiliary objective** ($L_3$) during fine-tuning helped **improve generalization and accelerate convergence**.<br>
        - **Minimal Changes**: Crucially, the **only extra parameters required during fine-tuning are $W_y$ and embeddings for delimiter tokens**.<br>
        
3. **Task-Specific Input Transformations**<br>
To apply the model to structured NLP tasks (like textual entailment or question answering) while maintaining the single, contiguous sequence input required by the pre-trained architecture, the authors employ **task-aware input transformations** derived from traversal-style approaches. These transformations **allow the same basic model architecture to handle diverse inputs with minimal changes**:
    - **Sentence Boundaries**: All input sequences are surrounded by randomly initialized start (<s\>) and end (<e\>) tokens.
    - **Textual Entailment** (Sentence Pairs): The premise (p) and hypothesis (h) are concatenated with a delimiter token ($) placed in between: [ $p$; $; $h$ ].
    - **Similarity** (Non-Ordered Pairs): To account for the lack of inherent ordering, the input **includes both possible sentence orderings** (each separated by a delimiter). The resulting sequence representations ($h^l_m$) from processing both orderings are then added **element-wise** before being fed into the final linear layer.
    - **Question Answering**: The **document context** (z), **question** (q), and **each possible answer** { $a_k$ } are concatenated, separated by delimiters: [ $z$; $q$; $; $a_k$ ]. Each potential answer sequence is processed independently, and the outputs are **normalized** via a **softmax layer** to predict the correct answer.

## Key Achievements
- Introduced a novel **two-stage training procedure** (Unsupervised Pre-training + Supervised Fine-tuning) to learn a universal representation that transfers effectively to diverse tasks.
- Use of **task-aware input transformations** during fine-tuning that converts structured inputs into a single contiguous sequence of tokens that the pre-trained Transformer model can process. It avoids the need for extensive, task-specific changes to the model architecture that previous transfer learning techniques often required.

## Pros & Cons

Pros
- Unified approach to NLP tasks
- Demonstrated transfer learning in NLP

Cons
- No bidirectional context as a **unidirectional** (left-to-right) model that could not use future tokens for context. It has a limited understanding in some tasks compared to BERT, which uses bidirectional attention.
- Required fine-tuning for each new task.
- Limited dataset, trained on BooksCorpus (≈7000 unpublished books), which was relatively small (~5GB of text) and not diverse enough for broad generalization.
- 
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

