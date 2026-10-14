import argparse
import sys
from pathlib import Path

import torch

# Running this file directly puts ``scripts/`` on sys.path, but the assignment
# package lives one level above it. Add that project root so the documented
# ``python scripts/run_generate.py`` command works without installing the repo
# or setting PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_basics.generate import decode  # noqa: E402
from cs336_basics.model import TransformerLanguageModel  # noqa: E402
from cs336_basics.tokenizer import Tokenizer  # noqa: E402


def default_device():
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() > 4:
        return "cuda:4"
    return "cuda"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a continuation using the OWT 32k A1 checkpoint."
    )
    parser.add_argument("--vocab_path", default="outputs/owt_bpe_32k.vocab.pkl")
    parser.add_argument("--merges_path", default="outputs/owt_bpe_32k.merges.pkl")
    parser.add_argument(
        "--checkpoint_path", default="outputs/checkpoints/owt_32k_latest.pt"
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default=default_device())

    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    return parser


def main():
    args = build_parser().parse_args()

    tokenizer = Tokenizer.from_files(
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    text = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print(text)


if __name__ == "__main__":
    main()
