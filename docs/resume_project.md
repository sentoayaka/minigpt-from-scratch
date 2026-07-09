# Resume Project Notes: MiniGPT from Scratch

## Project Name

MiniGPT from Scratch: Byte-Pair Tokenizer and Transformer Language Model

## One-Line Summary

Implemented a 22.70M-parameter GPT-style language model training stack from scratch in PyTorch, including BPE tokenization, Transformer blocks, AdamW training, validation logging, checkpointing, and autoregressive text generation.

## Technical Highlights

- Built a byte-pair encoding tokenizer with train, encode, decode, and iterable encoding support.
- Implemented core Transformer LM components from scratch: Linear, Embedding, RMSNorm, SwiGLU, RoPE, causal self-attention, Transformer block, and LM head.
- Implemented training utilities: cross-entropy loss, AdamW, cosine LR schedule with warmup, gradient clipping, mini-batch sampling, checkpoint save/load, and validation.
- Added experiment tooling: CSV loss logging, SVG loss plotting, scripted TinyStories training, CLI generation, and an interactive prompt loop.

## Concrete Data

```text
Dataset used locally          TinyStories
Vocabulary size               10,000
Context length                256
Model width                   512
Layers                        4
Attention heads               16
Feed-forward dimension        1,344
Parameter count               22,696,448, about 22.70M
Optimizer                     AdamW
Learning-rate schedule        warmup + cosine decay
Unit tests                    47 passed, 1 xpassed
Smoke run valid loss          9.2907 at step 0 -> 8.9374 at step 3
```

## Resume Bullets

- Implemented a 22.70M-parameter GPT-style Transformer language model from scratch in PyTorch, including BPE tokenization, RoPE causal self-attention, RMSNorm, SwiGLU MLP layers, AdamW, cosine LR scheduling, checkpointing, and autoregressive generation.
- Built a reproducible TinyStories training pipeline with memory-mapped token-id datasets, random mini-batch sampling, validation loss evaluation, CSV experiment logging, and SVG loss-curve visualization.
- Verified correctness against the CS336 test suite with 47 passing tests and demonstrated prompt-based generation from a locally trained TinyStories checkpoint.

## Interview Talking Points

- Why the LM forward returns logits instead of probabilities.
- How causal masking prevents attending to future tokens.
- Why RoPE is applied to queries and keys, not values.
- How BPE converts raw text into a compact integer sequence.
- What train loss, validation loss, and qualitative generations each tell you.
