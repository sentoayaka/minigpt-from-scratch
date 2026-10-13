import pickle
import time
import tracemalloc

from cs336_basics.bpe import train_bpe


def longest_token(vocab):
    token = max(vocab.values(), key=len)
    return token, token.decode("utf-8", errors="replace")


def run(input_path, vocab_size, out_prefix):
    tracemalloc.start()
    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        profile = True
    )

    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    token_bytes, token_text = longest_token(vocab)

    print("elapsed_seconds:", elapsed)
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


if __name__ == "__main__":
    run(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        out_prefix="owt_bpe_32k",
    )