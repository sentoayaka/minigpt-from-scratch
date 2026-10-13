import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from itertools import islice
import os
from pathlib import Path
import shutil

import numpy as np

from cs336_basics.tokenizer import Tokenizer


_WORKER_TOKENIZER = None


def _initialize_tokenizer_worker(vocab_path, merges_path, special_tokens):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_files(
        vocab_path, merges_path, special_tokens=special_tokens
    )


def _encode_text_batch(lines):
    token_ids = []
    for line in lines:
        token_ids.extend(_WORKER_TOKENIZER.encode(line))
    if token_ids and (
        min(token_ids) < 0 or max(token_ids) > np.iinfo(np.uint16).max
    ):
        raise ValueError("token IDs must fit in uint16")
    return np.asarray(token_ids, dtype=np.uint16).tobytes()


def _text_batches(input_path, chunk_chars):
    batch = []
    batch_chars = 0
    with Path(input_path).open(encoding="utf-8") as text_file:
        for line in text_file:
            batch.append(line)
            batch_chars += len(line)
            if batch_chars >= chunk_chars:
                yield batch
                batch = []
                batch_chars = 0
    if batch:
        yield batch


def _parallel_encoded_chunks(
    input_path,
    vocab_path,
    merges_path,
    special_tokens,
    workers,
    chunk_chars,
):
    batches = iter(_text_batches(input_path, chunk_chars))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_tokenizer_worker,
        initargs=(str(vocab_path), str(merges_path), special_tokens),
    ) as executor:
        pending = deque()
        max_pending = workers * 2
        for _ in range(max_pending):
            try:
                pending.append(executor.submit(_encode_text_batch, next(batches)))
            except StopIteration:
                break

        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(executor.submit(_encode_text_batch, next(batches)))
            except StopIteration:
                pass


def _write_encoded_chunks_to_npy(encoded_chunks, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.name + ".uint16.tmp")
    npy_path = output_path.with_name(output_path.name + ".npy.tmp")
    count = 0

    try:
        with raw_path.open("wb") as raw_file:
            for chunk in encoded_chunks:
                if len(chunk) % np.dtype(np.uint16).itemsize:
                    raise ValueError("encoded token bytes must align to uint16 values")
                raw_file.write(chunk)
                count += len(chunk) // np.dtype(np.uint16).itemsize

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


def write_ids_to_npy(token_ids, output_path, chunk_size=1_000_000):
    """Write a token stream to uint16 .npy format without keeping it in RAM."""
    def encoded_chunks():
        iterator = iter(token_ids)
        while chunk := list(islice(iterator, chunk_size)):
            if min(chunk) < 0 or max(chunk) > np.iinfo(np.uint16).max:
                raise ValueError("token IDs must fit in uint16")
            yield np.asarray(chunk, dtype=np.uint16).tobytes()

    return _write_encoded_chunks_to_npy(encoded_chunks(), output_path)


def tokenize_file(
    input_path,
    vocab_path,
    merges_path,
    output_path,
    special_tokens,
    workers=1,
    chunk_chars=1_000_000,
):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be at least 1")

    if workers > 1:
        chunks = _parallel_encoded_chunks(
            input_path,
            vocab_path,
            merges_path,
            special_tokens,
            workers,
            chunk_chars,
        )
        return _write_encoded_chunks_to_npy(chunks, output_path)

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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-chars", type=int, default=1_000_000)
    args = parser.parse_args()

    count = tokenize_file(
        args.input,
        args.vocab,
        args.merges,
        args.output,
        special_tokens=args.special_token or ["<|endoftext|>"],
        workers=args.workers,
        chunk_chars=args.chunk_chars,
    )
    print(f"wrote {count:,} token IDs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
