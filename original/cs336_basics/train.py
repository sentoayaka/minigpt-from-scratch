import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.model import TransformerLanguageModel
from cs336_basics.optim import AdamW, get_lr_cosine_schedule_with_warmup, gradient_clipping


def build_model(args):
    return TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )


def append_metrics(log_path: str | None, row: dict[str, int | float | str | None]) -> None:
    if log_path is None:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "split", "loss", "lr"]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(args):
    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r") if args.valid_data else None

    model = build_model(args)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    model.train()

    for step in range(args.num_steps):
        lr = get_lr_cosine_schedule_with_warmup(
            step,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        if args.max_grad_norm is not None:
            gradient_clipping(model.parameters(), args.max_grad_norm)

        optimizer.step()

        if step % args.log_every == 0:
            train_loss = loss.item()
            print(f"step {step} | loss {train_loss:.4f} | lr {lr:.6e}")
            append_metrics(args.log_path, {"step": step, "split": "train", "loss": train_loss, "lr": lr})

        if valid_data is not None and step % args.eval_every == 0:
            val_loss = evaluate(model, valid_data, args)
            print(f"step {step} | val_loss {val_loss:.4f}")
            append_metrics(args.log_path, {"step": step, "split": "valid", "loss": val_loss, "lr": lr})

        if args.checkpoint_path and step % args.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)

    if valid_data is not None:
        val_loss = evaluate(model, valid_data, args)
        print(f"step {args.num_steps} | val_loss {val_loss:.4f}")
        append_metrics(args.log_path, {"step": args.num_steps, "split": "valid", "loss": val_loss, "lr": None})

    if args.checkpoint_path:
        save_checkpoint(model, optimizer, args.num_steps, args.checkpoint_path)


@torch.no_grad()
def evaluate(model, valid_data, args):
    model.eval()

    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(valid_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    model.train()
    return np.mean(losses)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=1000)

    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
