# VFC FAC adaptation specification

This phase adapts Feature Activation Coverage from *Less is Enough* (arXiv
2602.10388v4) to the frozen Qwen3-VL-8B SFT policy. The paper and its official
implementation describe SAE feature coverage as a data-space diagnostic; they
do not define a VQG reward or a multimodal image/question scorer.

## Preserved from FAC

- tied-weight ReLU sparse autoencoder;
- Top-K sparse feature activation (`K=20`);
- max pooling over target-token activations;
- feature-frequency coverage and train-only anchor/policy frequency comparison;
- undercoverage weight `max(0, log((p_anchor+1e-6)/(p_policy+1e-6)))`.

## Adapted for this VLM

- hidden states come from the frozen Qwen3-VL-8B SFT LoRA policy;
- the target is the question-token span only, capped at the first 16 tokens;
- the primary layer is fixed before development scoring at
  `floor(36/2)=18`, with hidden size 4096;
- real versus same-size gray blank images define visual sensitivity;
- VFC and UVFC are sums over the unique active feature set of each question.

## Fixed feasibility deviation

The paper's released examples use a 65,536-feature dictionary. This feasibility
run fixes a 4,096-feature dictionary, Top-K 20, three AdamW epochs, batch size
512, and learning rate 1e-3 before reading any development labels. No Luna,
GPT-4o, SeeQ, support, or Flat result enters SAE training, feature selection,
or calibration.

The SFT train manifest is the sole source for discovery, visual calibration,
human-question anchor frequencies, and one-greedy-question-per-train-image
policy frequencies. Development candidates are read only after all weights are
frozen.

References: [arXiv:2602.10388](https://arxiv.org/abs/2602.10388) and the
[official FAC-Synthesis implementation](https://github.com/Zhongzhi660/FAC-Synthesis).
