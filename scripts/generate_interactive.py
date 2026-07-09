import argparse

import torch

from cs336_basics.generate import decode
from cs336_basics.model import TransformerLanguageModel
from cs336_basics.tokenizer import Tokenizer


def load_model(args):
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
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
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prompt a local Transformer LM interactively.")
    parser.add_argument("--vocab_path", default="outputs/tinystories_bpe_vocab.pkl")
    parser.add_argument("--merges_path", default="outputs/tinystories_bpe_merges.pkl")
    parser.add_argument("--checkpoint_path", default="outputs/tinystories_ckpt.pt")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    print("Enter a prompt. Press Ctrl-D or submit an empty prompt to exit.")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        print(
            decode(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
        )
        print()


if __name__ == "__main__":
    main()
