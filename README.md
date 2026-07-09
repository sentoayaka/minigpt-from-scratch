# MiniGPT from Scratch

A compact GPT-style language-modeling system implemented from scratch in PyTorch for CS336 Assignment 1. The project includes a byte-pair encoding tokenizer, Transformer language model, training loop, checkpointing, validation, and prompt-based generation.

## What Is Implemented

- Byte-pair encoding tokenizer training and encode/decode utilities.
- GPT-style Transformer LM with token embeddings, RoPE causal self-attention, RMSNorm, SwiGLU MLP blocks, and an output projection head.
- Cross-entropy language-modeling loss, AdamW, cosine learning-rate decay with warmup, and gradient clipping.
- Memory-mapped token-id datasets with random mini-batch sampling.
- Checkpoint save/load, validation loss evaluation, CSV training logs, SVG loss plotting, and autoregressive text generation.

## Repository Layout

```text
cs336_basics/          core tokenizer, model, optimizer, loss, data, train, generate code
scripts/               runnable experiment and demo entrypoints
tests/                 CS336 unit tests and adapters
outputs/               local experiment outputs; large artifacts are git-ignored
data/                  local raw datasets; git-ignored
```

## Setup

```bash
cd /home/sentoayaka/cs336/assignment1-basics
.venv/bin/python -m pytest -q
```

The original course template recommends `uv run pytest`; this checkout also works with the local `.venv`.

## Data Pipeline

Download TinyStories/OpenWebText into `data/`, train BPE tokenizers with `scripts/run_bpe_experiment.py`, and tokenize data with `scripts/run_tokenizer_experiments.py`. In this local setup, the TinyStories tokenizer and token-id arrays are expected at:

```text
outputs/tinystories_bpe_vocab.pkl
outputs/tinystories_bpe_merges.pkl
outputs/tinystories_train_ids.npy
outputs/tinystories_valid_ids.npy
```

Large datasets, tokenizer pickle files, tokenized arrays, and checkpoints are intentionally ignored by git.

## Train TinyStories

A quick smoke run:

```bash
.venv/bin/python scripts/run_tinystories_train.py \
  --checkpoint_path outputs/tinystories_smoke_ckpt.pt \
  --log_path outputs/logs/tinystories_smoke.csv \
  --num_steps 200 \
  --warmup_iters 20 \
  --cosine_cycle_iters 200 \
  --eval_every 50 \
  --save_every 100
```

A longer experiment:

```bash
.venv/bin/python scripts/run_tinystories_train.py \
  --checkpoint_path outputs/tinystories_experiment_ckpt.pt \
  --log_path outputs/logs/tinystories_experiment.csv \
  --num_steps 5000 \
  --warmup_iters 500 \
  --cosine_cycle_iters 5000 \
  --eval_every 500 \
  --eval_iters 20 \
  --save_every 1000
```

## Plot Training Curve

```bash
.venv/bin/python scripts/plot_training_curve.py \
  --log_path outputs/logs/tinystories_experiment.csv \
  --output_path outputs/figures/tinystories_loss.svg
```

## Generate Text

Single prompt:

```bash
.venv/bin/python scripts/run_generate.py \
  --checkpoint_path outputs/tinystories_ckpt.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 120
```

Interactive prompt loop:

```bash
.venv/bin/python scripts/generate_interactive.py \
  --checkpoint_path outputs/tinystories_ckpt.pt
```

## Default TinyStories Model

```text
vocab_size      10,000
context_length  256
d_model         512
num_layers      4
num_heads       16
d_ff            1,344
parameters      22.70M
optimizer       AdamW
lr schedule     warmup + cosine decay
```

## Current Local Results

- Unit tests: `47 passed, 1 xpassed`.
- Smoke training run: 3 steps on TinyStories token ids, `batch_size=2`, used to verify the training loop and CSV logging.
- Checkpoint evaluation: `outputs/tinystories_ckpt_5k.pt` at iteration 4000 reaches validation loss `2.0899` over 50 eval batches, with perplexity `8.08`.
- CSV logs: `outputs/logs/tinystories_smoke.csv` and `outputs/logs/tinystories_eval.csv`.
- Loss curve: `outputs/figures/tinystories_smoke_loss.svg`.
- Example generation with an existing TinyStories checkpoint:

```text
Once upon a time, there was a little girl named Lily. She loved to help her mom in the park...
```

## Portfolio Summary

Implemented a GPT-style Transformer language model from scratch in PyTorch, including BPE tokenization, RoPE causal self-attention, RMSNorm, SwiGLU feed-forward layers, AdamW optimization, cosine learning-rate scheduling, checkpointing, and autoregressive text generation. Built a reproducible TinyStories training pipeline with validation loss tracking and prompt-based qualitative evaluation.
