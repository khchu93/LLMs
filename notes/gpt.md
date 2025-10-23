# GPT

## Motivation

The motivation for the paper stems from the pervasive dependence of modern Natural Language Processing (NLP) deep learning methods on substantial amounts of manually labeled data, which severely restricts their applicability, especially in domains lacking annotated resources. While previous approaches have successfully leveraged unsupervised learning, primarily through the use of pre-trained word embeddings, these methods mainly transfer only word-level information. Leveraging more than word-level semantics from unlabeled text is challenging for existing semi-supervised approaches, largely because there is no consensus on the most effective optimization objectives for learning text representations useful for transfer (with various methods like language modeling, machine translation, and discourse coherence showing mixed results), nor is there consensus on the best way to transfer those representations to target tasks without requiring substantial task-specific architectural modifications. Existing transfer techniques often involve adding intricate learning schemes, auxiliary objectives, or task-specific changes to the model architecture.

To address the scarcity of labeled data and the uncertainties surrounding effective transfer learning, the authors explore a semi-supervised approach utilizing a two-stage training procedure: unsupervised pre-training followed by supervised fine-tuning. The central goal is to learn a universal representation that can transfer with minimal adaptation to a wide range of language understanding tasks. The authors propose using a generative pre-training of a language model on a diverse corpus of unlabeled text to learn the initial parameters of a neural network. For the model architecture, they employ the Transformer decoder network, which provides a more structured memory for handling long-term dependencies compared to alternatives like recurrent networks (e.g., LSTMs). Crucially, during the discriminative fine-tuning stage, the authors introduce task-aware input transformations that convert structured text inputs (like sentence pairs for textual entailment or question/answer triplets) into a single contiguous token sequence. This approach minimizes changes to the model architecture during transfer, allowing the same general task-agnostic model to significantly improve the state of the art in 9 out of 12 studied tasks, demonstrating the effectiveness of the semi-supervised framework.

## Architecture
<img src="https://github.com/khchu93/NoteImage/blob/main/gpt.PNG" alt="embedding" width="600"/><br>

## Key Achievements
- 

## Pros & Cons

Pros
- 

Cons
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

