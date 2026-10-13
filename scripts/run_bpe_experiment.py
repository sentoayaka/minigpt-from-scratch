import argparse
import pickle
import time

from cs336_basics.bpe import train_bpe


def longest_token(vocab):
    token = max(vocab.values(), key=len)
    return token, token.decode("utf-8", errors="replace")


def run(input_path, vocab_size, out_prefix, trace_memory=False):
    tracer = None
    if trace_memory:
        import tracemalloc

        tracer = tracemalloc
        tracer.start()

    start = time.perf_counter()
    try:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            profile=True,
        )
        elapsed = time.perf_counter() - start
        if tracer is not None:
            _, peak = tracer.get_traced_memory()
    finally:
        if tracer is not None:
            tracer.stop()

    token_bytes, token_text = longest_token(vocab)

    print("elapsed_seconds:", elapsed)
    if tracer is not None:
        print("peak_memory_gb:", peak / 1024**3)
    print("vocab_size:", len(vocab))
    print("num_merges:", len(merges))
    print("longest_token_bytes:", token_bytes)
    print("longest_token_text:", repr(token_text))
    print("longest_token_length:", len(token_bytes))

    with open(out_prefix + ".vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(out_prefix + ".merges.pkl", "wb") as f:
        pickle.dump(merges, f)


def main():
    parser = argparse.ArgumentParser(description="Train and profile the assignment BPE tokenizer.")
    parser.add_argument("--input-path", default="data/owt_train.txt")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--out-prefix", default="owt_bpe_32k")
    parser.add_argument(
        "--trace-memory",
        action="store_true",
        help="enable tracemalloc (slows BPE training substantially)",
    )
    args = parser.parse_args()
    run(args.input_path, args.vocab_size, args.out_prefix, trace_memory=args.trace_memory)


if __name__ == "__main__":
    main()
