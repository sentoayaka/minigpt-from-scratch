import argparse
from argparse import Namespace

import torch

from cs336_basics.train import train


def main():
    parser = argparse.ArgumentParser(description="Train the default TinyStories Transformer LM experiment.")
    parser.add_argument("--train_data", default="outputs/tinystories_train_ids.npy")
    parser.add_argument("--valid_data", default="outputs/tinystories_valid_ids.npy")
    parser.add_argument("--checkpoint_path", default="outputs/tinystories_experiment_ckpt.pt")
    parser.add_argument("--log_path", default="outputs/logs/tinystories_experiment.csv")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    train(
        Namespace(
            train_data=args.train_data,
            valid_data=args.valid_data,
            checkpoint_path=args.checkpoint_path,
            log_path=args.log_path,
            device=args.device,
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            d_ff=1344,
            rope_theta=10000.0,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.01,
            max_grad_norm=1.0,
            log_every=args.log_every,
            eval_every=args.eval_every,
            eval_iters=args.eval_iters,
            save_every=args.save_every,
        )
    )


if __name__ == "__main__":
    main()
