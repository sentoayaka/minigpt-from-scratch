# 中文简历项目：MiniGPT from Scratch

## 简历项目名称

从零实现小型 GPT 语言模型训练系统

## 简历一行版

从零实现一个 22.70M 参数的 GPT 风格语言模型训练系统，覆盖 BPE 分词器、Transformer Decoder、AdamW 训练、验证集评估、checkpoint 保存和自回归文本生成。

## 简历条目版

**从零实现小型 GPT 语言模型训练系统 | PyTorch, Transformer, BPE, TinyStories**

- 从零实现 GPT 风格 Transformer Language Model，包含 BPE tokenizer、Token Embedding、RoPE 因果自注意力、RMSNorm、SwiGLU FFN、LM Head 和自回归生成流程。
- 搭建完整训练流水线，支持文本语料 BPE 训练、token ids 数据集构建、mini-batch 采样、cross-entropy loss、AdamW 优化器、warmup + cosine learning rate schedule、gradient clipping 和 checkpoint 保存。
- 在 TinyStories 数据集上完成本地训练与验证实验，默认模型规模为 22.70M 参数，配置为 vocab size 10,000、context length 256、d_model 512、4 layers、16 heads、d_ff 1,344。
- 补充工程化实验工具，包括训练 loss CSV 记录、SVG loss 曲线绘制、命令行文本生成脚本和交互式 prompt 生成脚本，便于复现实验与展示模型效果。
- 通过 CS336 assignment 测试集验证核心实现正确性，测试结果为 47 passed、1 xpassed；基于 TinyStories checkpoint 评估得到 validation loss 2.0899，perplexity 8.08。

## 更短的简历版

- 从零实现 22.70M 参数 GPT 风格语言模型训练系统，包含 BPE tokenizer、RoPE causal self-attention、RMSNorm、SwiGLU、AdamW、学习率调度、checkpoint 和自回归文本生成。
- 搭建 TinyStories 训练与评估流水线，支持 tokenization、mini-batch 采样、validation loss 评估、CSV 日志记录和 SVG loss 曲线可视化；TinyStories checkpoint validation loss 为 2.0899，测试结果为 47 passed、1 xpassed。

## 项目数据

```text
数据集                  TinyStories
词表大小                10,000
上下文长度              256
模型宽度                512
Transformer 层数        4
注意力头数              16
FFN 维度                1,344
参数量                  22,696,448，约 22.70M
优化器                  AdamW
学习率策略              warmup + cosine decay
测试结果                47 passed, 1 xpassed
checkpoint valid loss   2.0899
checkpoint perplexity   8.08
smoke run valid loss    9.2907 -> 8.9374（仅用于验证训练 loop）
```

## 面试展开说法

这个项目不是调用现成的 HuggingFace 模型，而是从底层模块开始实现一个 GPT 风格语言模型。数据侧先训练 BPE tokenizer，把原始文本转成 token ids；模型侧实现 decoder-only Transformer，用 RoPE 提供位置信息，用 causal mask 保证只能看历史 token；训练侧实现 cross-entropy loss、AdamW、学习率调度和 checkpoint；推理侧按 prompt 逐 token 采样生成文本。

项目的重点是把“模型结构、训练流程、实验记录和生成 demo”串成一个完整系统，而不只是单独写几个模块。最后可以通过 loss 曲线和 prompt 生成结果观察模型是否真的学到了 TinyStories 的文本分布。

## 可展示命令

```bash
python scripts/run_tinystories_train.py
python scripts/plot_training_curve.py
python scripts/run_generate.py --prompt "Once upon a time"
python scripts/generate_interactive.py
```