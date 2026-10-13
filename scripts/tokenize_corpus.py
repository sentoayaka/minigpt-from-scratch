import argparse
from itertools import islice
import os
from pathlib import Path
import shutil

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def write_ids_to_npy(token_ids, output_path, chunk_size=1_000_000):
    """Write a token stream to uint16 .npy format without keeping it in RAM."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.name + ".uint16.tmp")
    npy_path = output_path.with_name(output_path.name + ".npy.tmp")
    count = 0

    try:
        iterator = iter(token_ids)
        with raw_path.open("wb") as raw_file:
            while chunk := list(islice(iterator, chunk_size)):
                if min(chunk) < 0 or max(chunk) > np.iinfo(np.uint16).max:
                    raise ValueError("token IDs must fit in uint16")
                array = np.asarray(chunk, dtype=np.uint16)
                array.tofile(raw_file)
                count += array.size

        if count == 0:
            raise ValueError("cannot write an empty token stream")

        header = {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.uint16)),
            "fortran_order": False,
            "shape": (count,),
        }
        with npy_path.open("wb") as npy_file:
            np.lib.format.write_array_header_1_0(npy_file, header)
            with raw_path.open("rb") as raw_file:
                shutil.copyfileobj(raw_file, npy_file, length=8 * 1024 * 1024)
        os.replace(npy_path, output_path)
    finally:
        raw_path.unlink(missing_ok=True)
        npy_path.unlink(missing_ok=True)

    return count


def tokenize_file(input_path, vocab_path, merges_path, output_path, special_tokens):
    tokenizer = Tokenizer.from_files(
        vocab_path, merges_path, special_tokens=special_tokens
    )
    with Path(input_path).open(encoding="utf-8") as text_file:
        return write_ids_to_npy(
            tokenizer.encode_iterable(text_file), output_path=output_path
        )


def main():
    parser = argparse.ArgumentParser(description="Stream a text corpus into uint16 token IDs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--special-token", action="append", default=[])
    args = parser.parse_args()

    count = tokenize_file(
        args.input,
        args.vocab,
        args.merges,
        args.output,
        special_tokens=args.special_token or ["<|endoftext|>"],
    )
    print(f"wrote {count:,} token IDs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
