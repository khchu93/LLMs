# DeepSeek

## Motivation

**Current Models' Limitations**<br>
Current LLMs have been undergoing rapid iteration and evolution, progressively diminishing the gap toward `Artificial General Intelligence (AGI)`. Post-training techniques have emerged as crucial for enhancing accuracy on reasoning tasks, alignment with social values, and adaptability to user preferences, often requiring minimal computational resources compared to pre-training. Specifically regarding reasoning, OpenAI’s o1 series models have introduced inference-time scaling by increasing the length of the `Chain-of-Thought (CoT)` reasoning process, achieving significant improvements in mathematics, coding, and scientific reasoning. 

However, despite this success, the challenge of **effective test-time scaling** remains an open question. Furthermore, prior work exploring various methods like process-based reward models, `reinforcement learning (RL)`, and search algorithms such as `Monte Carlo Tree Search (MCTS)` has **failed to achieve general reasoning performance comparable to OpenAI’s o1 series models**. Additionally, existing RL approaches often **heavily depend on supervised data** (Supervised Fine-Tuning, or SFT), which is **time-intensive to gather**.

**Proposed Solution**<br>
To address these limitations and improve language model reasoning capabilities, the authors propose an approach centered on **pure `reinforcement learning (RL)`** and a novel **multi-stage training pipeline**. The first step in their exploration is `DeepSeek-R1-Zero`, a model trained via **large-scale RL without relying on SFT** as a preliminary step, intended to explore the potential of LLMs to develop reasoning capabilities through self-evolution. DeepSeek-R1-Zero remarkably demonstrates capabilities like **self-verification, reflection, and generating long CoTs** purely through RL, marking a significant milestone by validating that **reasoning can be incentivized without SFT data**. Despite DeepSeek-R1-Zero's success in developing powerful reasoning behaviors, it faced practical challenges such as **poor readability and language mixing**. To overcome these drawbacks and further enhance performance, the authors introduce `DeepSeek-R1`, which incorporates a small amount of **"cold-start" data** (high-quality, human-friendly long CoT examples) and a **multi-stage training pipeline** featuring **two RL stages and two SFT stages**. This approach aims to create a user-friendly model that produces clear and coherent CoTs while achieving performance comparable to OpenAI-o1-1217 on reasoning tasks.

## Architecture

The `DeepSeek-R1` series represents a post-training innovation built on the powerful `DeepSeek-V3-Base` foundation. DeepSeek-R1-Zero serves as the initial research model, designed to explore the inherent potential of language models to develop reasoning capabilities solely through large-scale `Reinforcement Learning (RL)`, without relying on `Supervised Fine-Tuning (SFT)` data as a prerequisite. This experiment validated that complex reasoning behaviors, such as self-verification and the generation of long `Chains-of-Thought (CoTs)`, can emerge autonomously under pure RL. However, `R1-Zero` suffered from practical deficiencies like poor readability and language mixing. `DeepSeek-R1` is the refined production model, which incorporates a multi-stage training pipeline starting with "cold-start" SFT data to ensure user-friendly output and coherence, ultimately achieving performance comparable to leading closed-source models like `OpenAI-o1-1217` on reasoning benchmarks.

**The Core Base Model: `DeepSeek-V3-Base`**<br>
Both R1 models rely on `DeepSeek-V3-Base`, a large language model utilizing a `Mixture-of-Experts (MoE)` Transformer architecture. `DeepSeek-V3` operates with a substantial scale of 671 billion total parameters, while maintaining efficiency by activating only 37 billion parameters per token. The core architectural innovations adopted from `DeepSeek-V2` and utilized here are `DeepSeekMoE` for efficient training and `Multi-head Latent Attention (MLA)` for efficient inference.

`DeepSeekMoE` and Auxiliary-Loss-Free Load Balancing<br>
The `Feed-Forward Networks (FFNs)` in the Transformer blocks (except for the first three layers) are implemented using `DeepSeekMoE`. In `DeepSeek-V3`, each `MoE` layer consists of 1 shared expert and 256 routed experts, with 8 routed experts activated for each token.
To manage the inherent complexity of balancing expert usage in `MoE` models, `DeepSeek-V3` pioneers an auxiliary-loss-free load balancing strategy, overcoming the performance degradation associated with traditional auxiliary loss methods. This strategy works by introducing a bias term ($b_i$) for each expert; this bias is dynamically adjusted (increased for underloaded experts, decreased for overloaded experts) to influence the routing decisions, which are based on the token-to-expert affinity score $s_{j,t}$ plus the bias $b_j(s_{j,t}+b_j)$. Crucially, the bias term is only used for routing; the final gating value multiplied with the FFN output is still derived from the original affinity score $s_{i,t}$. This technique maintains a balanced expert load throughout training while maximizing model performance, fostering greater expert specialization compared to auxiliary-loss-based models.

`Multi-head Latent Attention (MLA)`<br>
For the attention mechanism, DeepSeek-V3 uses `Multi-head Latent Attention (MLA)` to drastically reduce the memory overhead during inference. The functionality centers on low-rank joint compression of the attention **keys** ($K$) and **values** ($V$). The input hidden state is projected down to a much smaller **compressed latent vector ($c_{KV_t}$) **, which, along with a decoupled key that carries `Rotary Positional Embedding (RoPE)`, is cached during the generation phase. By compressing the $K/V$ cache dimensions ($d_c$ is much smaller than $d_hn_h$), `MLA` achieves comparable performance to standard `Multi-Head Attention (MHA)` while significantly reducing the memory footprint. Furthermore, the attention **queries** ($Q$) are also compressed during training to reduce activation memory.

**`DeepSeek-R1-Zero`: Pure RL Implementation**<br>
`DeepSeek-R1-Zero` implements a pure Reinforcement Learning approach on the `DeepSeek-V3-Base` model. To manage computational costs inherent in RL, it employs `Group Relative Policy Optimization (GRPO)`, which is crucial because it foregoes the need for a critic model (often the same size as the policy model) by estimating the training baseline from group scores instead. The training uses a rule-based reward system consisting of accuracy rewards (verifying correct answers, often in a specified format) and format rewards (enforcing the reasoning process to be enclosed in <think> and </think> tags). This approach successfully validated that RL alone could drive substantial performance gains (e.g., AIME 2024 Pass@1 rose from 15.6% to 71.0%) and led to the spontaneous emergence of complex behaviors like reflection and long CoT generation. The major engineering drawback was the resulting poor readability and language mixing.

**`DeepSeek-R1`: Multi-Stage Refinement**<br>
`DeepSeek-R1` addresses the practical issues of `R1-Zero` via a sophisticated multi-stage pipeline.
1. Cold Start (`SFT` Stage 1): Training begins with a small amount of high-quality, human-friendly long `CoT` data to provide a readable foundation and stabilize the initial model, ensuring responses adhere to a clear pattern (e.g., including a summary alongside the reasoning process).
2. Reasoning-oriented `RL` (`RL` Stage 1): Large-scale `GRPO` is applied to the cold-start checkpoint, focusing on domains like math and code. To solve `R1-Zero`'s issue, a language consistency reward is introduced. This reward, calculated as the proportion of target language words in the `CoT`, is directly summed with the accuracy reward to align the model with human preferences for readable output, despite causing a slight performance degradation in initial ablation tests.
3. Rejection Sampling and `SFT` (`SFT` Stage 2): After `RL` convergence, approximately 800k new `SFT` samples are curated. Reasoning trajectories (∼600k samples) are generated using the converged `RL` model, employing rejection sampling to filter chaotic or mixed-language `CoTs`. This stage is supplemented with ∼200k samples from `DeepSeek-V3`'s existing `SFT` pipeline for general non-reasoning domains (writing, QA) to enhance overall capabilities.
4. `RL` for All Scenarios (RL Stage 2): A final `RL` stage is implemented for comprehensive human preference alignment (helpfulness and harmlessness). This stage uses rule-based rewards for reasoning tasks and reward models (often `DeepSeek-V3` itself performing judgment) for general, nuanced scenarios. Helpfulness is assessed primarily on the final summary to avoid interfering with the detailed underlying reasoning process.
This methodical approach yields `DeepSeek-R1`, which retains the powerful emergent reasoning features discovered in `R1-Zero` while providing human-readable, high-performance outputs.

## Key Achievements
- Resolved the challenges of poor readability and language mixing encountered by DeepSeek-R1-Zero. This was accomplished using a multi-stage training pipeline that introduced "cold-start" SFT data and a language consistency reward in the RL stage.
- Validated that LLM reasoning capabilities can be incentivized purely through Reinforcement Learning (RL), without requiring Supervised Fine-Tuning (SFT) as a preliminary step.
- Induced emergent cognitive behaviours such as self-reflection and iterative reasoning.

## Pros & Cons

Pros
- Pure RL-based reasoning capability
    - Showed that reinforcement learning alone (without SFT) can elicit reasoning behaviors, marking a major research breakthrough.
- Emergent self-reflection and iterative reasoning
    - The model spontaneously learned to reflect, revise, and extend its own reasoning chains, indicating a more “meta-cognitive” process.
- Effective knowledge distillation
    - Successfully transferred reasoning ability into smaller models (1.5B–70B), making high-quality reasoning more accessible.

Cons
- High computational and training complexity
    - Despite optimizations, RL-based reasoning tuning still requires large-scale compute and long training cycles, limiting replication by smaller labs.
- Reward design sensitivity
    - The quality of reasoning heavily depends on reward function design (e.g., correctness, readability, consistency).

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
[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)<br>
[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
