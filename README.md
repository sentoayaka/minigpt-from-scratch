# MiniGPT from Scratch

中文 | [English](#english-version)

## 项目简介

这是一个基于 Stanford CS336 Assignment 1 整理出来的 GPT 风格语言模型项目。我目前完成的是 assignment 1 的核心内容，因此这个公开仓库定位为一个“小型 GPT from scratch”作品集：从 BPE tokenizer 到 Transformer 语言模型、训练循环、checkpoint、验证日志和文本生成，尽量把大模型训练的最小闭环做清楚。

## 实现内容

- BPE tokenizer 训练、encode/decode 与数据预处理流程。
- GPT 风格 Transformer LM：token embedding、RoPE causal self-attention、RMSNorm、SwiGLU MLP、输出投影层。
- Cross-entropy 语言建模损失、AdamW、warmup + cosine decay 学习率调度、gradient clipping。
- 基于 memory-mapped token id 的随机 mini-batch 采样。
- Checkpoint 保存/加载、validation loss 评估、CSV 训练日志、loss 曲线绘制和自回归文本生成。

## 仓库结构

```text
cs336_basics/          tokenizer、model、optimizer、loss、data、train、generate 等核心代码
scripts/               可运行的实验脚本和 demo 入口
tests/                 CS336 单元测试与适配代码
outputs/               本地实验输出；大文件已通过 .gitignore 排除
data/                  本地原始数据；不纳入 Git
```

## 运行方式

```bash
python -m pytest -q
```

训练 TinyStories 的 smoke run：

```bash
python scripts/run_tinystories_train.py \
  --checkpoint_path outputs/tinystories_smoke_ckpt.pt \
  --log_path outputs/logs/tinystories_smoke.csv \
  --num_steps 200 \
  --warmup_iters 20 \
  --cosine_cycle_iters 200 \
  --eval_every 50 \
  --save_every 100
```

生成文本：

```bash
python scripts/run_generate.py \
  --checkpoint_path outputs/tinystories_ckpt.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 120
```

## 默认 TinyStories 配置

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

## 本地结果

- 单元测试：`47 passed, 1 xpassed`。
- TinyStories checkpoint 评估：iteration 4000 的 validation loss 为 `2.0899`，perplexity 为 `8.08`。
- OWT 32k A1 训练：4 卡 DDP（GPU 4–7），20,000 步，最终 validation loss `4.2499`，最佳 `4.2172`（step 18,000）。完整参数、机器信息和耗时口径见[训练记录](docs/OWT训练与结果.md)；仓库保存了轻量 CSV 和 SVG。
- 大模型 checkpoint、OWT 原始数据、tokenizer pickle 和 tokenized arrays 留在训练服务器，没有上传。
- 示例生成：

```text
Once upon a time, there was a little girl named Lily. She loved to help her mom in the park...
```

## 为什么放在 GitHub 主页

这个项目展示的是我对大模型训练最小闭环的理解：tokenizer、Transformer 结构、优化器、学习率调度、数据采样、训练日志、验证和生成。它不是调用现成模型 API 的 demo，而是从底层模块开始实现一个可训练的小型语言模型。

## English Version

MiniGPT from Scratch is a compact GPT-style language-modeling project based on Stanford CS336 Assignment 1. Since my current CS336 implementation covers Assignment 1, this public repository is positioned as a small but complete "GPT from scratch" portfolio project: BPE tokenization, Transformer language modeling, training, checkpointing, validation, logging, and autoregressive generation.

## What Is Implemented

- BPE tokenizer training, encoding/decoding, and preprocessing utilities.
- GPT-style Transformer LM with token embeddings, RoPE causal self-attention, RMSNorm, SwiGLU MLP blocks, and an output projection head.
- Cross-entropy language-modeling loss, AdamW, warmup + cosine learning-rate decay, and gradient clipping.
- Memory-mapped token-id datasets with random mini-batch sampling.
- Checkpoint save/load, validation loss evaluation, CSV logs, loss plotting, and text generation.

## Layout

```text
cs336_basics/          core tokenizer, model, optimizer, loss, data, train, generate code
scripts/               runnable experiment and demo entrypoints
tests/                 CS336 unit tests and adapters
outputs/               local experiment outputs; large artifacts are git-ignored
data/                  local raw datasets; git-ignored
```

## Local Results

- Unit tests: `47 passed, 1 xpassed`.
- TinyStories checkpoint evaluation: validation loss `2.0899` at iteration 4000, perplexity `8.08`.
- The OWT 32k A1 run completed 20,000 steps on four GPUs (physical GPU 4–7); final validation loss was `4.2499`, with a best of `4.2172` at step 18,000. See the [training report](docs/OWT训练与结果.md), committed metrics CSV, and validation-loss SVG.
- Large checkpoints, raw OWT data, tokenizer pickles, and tokenized arrays remain on the training server and are excluded from Git.

## Portfolio Summary

This project demonstrates my understanding of the minimal training loop for language models: tokenization, Transformer architecture, optimization, scheduling, data sampling, logging, validation, and generation. It is a from-scratch learning project rather than a wrapper around an existing model API.
