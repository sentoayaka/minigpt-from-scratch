from pathlib import Path
import time
import numpy as np

from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

def load_tokenizer(name):
    vocab_path = OUT_DIR / f"{name}_vocab.pkl"
    merges_path = OUT_DIR / f"{name}_merges.pkl"
    return Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=["<|endoftext|>"],
    )

def sample_documents(path, n=10, sep="<|endoftext|>"):
    text = path.read_text(encoding="utf-8")
    docs = [doc for doc in text.split(sep) if doc.strip()]
    return docs[:n]

def compression_ratio(tokenizer, docs):
    total_bytes = 0
    total_tokens = 0

    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))

    return total_bytes / total_tokens

def throughput(tokenizer, docs):
    text = "".join(docs)
    total_bytes = len(text.encode("utf-8"))

    start = time.perf_counter()
    ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start

    return total_bytes / elapsed, len(ids), elapsed

def encode_to_uint16(tokenizer, input_path, output_path):
    ids = list(tokenizer.encode_iterable(input_path.open(encoding="utf-8")))
    arr = np.array(ids, dtype=np.uint16)
    np.save(output_path, arr)


def main():
    tinystories_tokenizer = load_tokenizer("tinystories_bpe")
    owt_tokenizer = load_tokenizer("owt_bpe")

    tinystories_docs = sample_documents(DATA_DIR / "TinyStoriesV2-GPT4-valid.txt")
    owt_docs = sample_documents(DATA_DIR / "owt_valid.txt")

    print("compression ratios:")
    print("TinyStories tokenizer on TinyStories:", compression_ratio(tinystories_tokenizer, tinystories_docs))
    print("OWT tokenizer on OWT:", compression_ratio(owt_tokenizer, owt_docs))
    print("TinyStories tokenizer on OWT:", compression_ratio(tinystories_tokenizer, owt_docs))

    bps, n_tokens, elapsed = throughput(tinystories_tokenizer, tinystories_docs)
    pile_seconds = 825 * 1024**3 / bps
    print("throughput bytes/sec:", bps)
    print("tokens:", n_tokens)
    print("elapsed:", elapsed)
    print("estimated Pile tokenize hours:", pile_seconds / 3600)

    # 需要真正保存训练用 token ids 时再打开：
    encode_to_uint16(
        tinystories_tokenizer,
        DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
        OUT_DIR / "tinystories_train_ids.npy",
    )


if __name__ == "__main__":
    main()