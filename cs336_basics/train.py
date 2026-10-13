import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.model import TransformerLanguageModel
from cs336_basics.optim import AdamW, get_lr_cosine_schedule_with_warmup, gradient_clipping


class CudaBatchPrefetcher:
    """Prepare the next CPU batch while the current step runs on the GPU."""

    def __init__(self, dataset, batch_size, context_length, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.next_batch = None
        self.preload()

    def preload(self):
        x_cpu, y_cpu = get_batch(
            self.dataset, self.batch_size, self.context_length, "cpu"
        )
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
        with torch.cuda.stream(self.stream):
            self.next_batch = (
                x_cpu.to(self.device, non_blocking=True),
                y_cpu.to(self.device, non_blocking=True),
            )

    def next(self):
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)
        batch = self.next_batch
        for tensor in batch:
            tensor.record_stream(current_stream)
        self.preload()
        return batch


def _init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    owns_process_group = False
    if world_size > 1 and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        dist.init_process_group(backend=backend)
        owns_process_group = True

    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    return owns_process_group, distributed, rank, world_size


def _make_batcher(dataset, args):
    if str(args.device).startswith("cuda"):
        return CudaBatchPrefetcher(
            dataset, args.batch_size, args.context_length, args.device
        )
    return None


def _next_batch(dataset, args, batcher):
    if batcher is not None:
        return batcher.next()
    return get_batch(dataset, args.batch_size, args.context_length, args.device)


def _save_checkpoint_if_primary(model, optimizer, iteration, checkpoint_path, rank):
    if rank == 0 and checkpoint_path:
        checkpoint_model = model.module if isinstance(model, DistributedDataParallel) else model
        save_checkpoint(checkpoint_model, optimizer, iteration, checkpoint_path)
    if dist.is_initialized():
        dist.barrier()


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
    owns_process_group, distributed, rank, world_size = _init_distributed(args)
    try:
        return _train(args, distributed, rank, world_size)
    finally:
        if owns_process_group:
            dist.destroy_process_group()


def _train(args, distributed, rank, world_size):
    seed = getattr(args, "seed", 0) + rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r") if args.valid_data else None

    model = build_model(args)
    if distributed:
        device_ids = [torch.cuda.current_device()] if str(args.device).startswith("cuda") else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    model.train()
    train_batcher = _make_batcher(train_data, args)

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

        x, y = _next_batch(train_data, args, train_batcher)

        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        if args.max_grad_norm is not None:
            gradient_clipping(model.parameters(), args.max_grad_norm)

        optimizer.step()

        if step % args.log_every == 0 and rank == 0:
            train_loss = loss.item()
            print(f"step {step} | loss {train_loss:.4f} | lr {lr:.6e}", flush=True)
            append_metrics(args.log_path, {"step": step, "split": "train", "loss": train_loss, "lr": lr})

        if valid_data is not None and step % args.eval_every == 0:
            val_loss = evaluate(model, valid_data, args, world_size=world_size)
            if rank == 0:
                print(f"step {step} | val_loss {val_loss:.4f}", flush=True)
                append_metrics(args.log_path, {"step": step, "split": "valid", "loss": val_loss, "lr": lr})

        if args.checkpoint_path and step % args.save_every == 0 and step > 0:
            _save_checkpoint_if_primary(model, optimizer, step, args.checkpoint_path, rank)

    if valid_data is not None:
        val_loss = evaluate(model, valid_data, args, world_size=world_size)
        if rank == 0:
            print(f"step {args.num_steps} | val_loss {val_loss:.4f}", flush=True)
            append_metrics(args.log_path, {"step": args.num_steps, "split": "valid", "loss": val_loss, "lr": None})

    if args.checkpoint_path:
        _save_checkpoint_if_primary(model, optimizer, args.num_steps, args.checkpoint_path, rank)


@torch.no_grad()
def evaluate(model, valid_data, args, world_size=1):
    model.eval()

    losses = []
    batcher = _make_batcher(valid_data, args)
    for _ in range(args.eval_iters):
        x, y = _next_batch(valid_data, args, batcher)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    model.train()
    mean_loss = float(np.mean(losses))
    if dist.is_initialized():
        mean_tensor = torch.tensor(mean_loss, dtype=torch.float64, device=args.device)
        dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
        mean_loss = (mean_tensor / world_size).item()
    return mean_loss


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
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=20000)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5000)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
