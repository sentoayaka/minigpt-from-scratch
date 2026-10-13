import numpy as np
import pickle
import pytest

from scripts.tokenize_corpus import tokenize_file, write_ids_to_npy


def test_write_ids_to_npy_streams_uint16_chunks(tmp_path):
    output_path = tmp_path / "tokens.npy"

    count = write_ids_to_npy(iter([1, 255, 256, 32767, 65535]), output_path, chunk_size=2)

    actual = np.load(output_path)
    assert count == 5
    assert actual.dtype == np.uint16
    assert actual.tolist() == [1, 255, 256, 32767, 65535]
    assert not (tmp_path / "tokens.npy.uint16.tmp").exists()


@pytest.mark.parametrize("token_ids", [[], [-1], [65536]])
def test_write_ids_to_npy_rejects_invalid_stream(tmp_path, token_ids):
    with pytest.raises(ValueError):
        write_ids_to_npy(token_ids, tmp_path / "invalid.npy", chunk_size=2)


def test_tokenize_file_uses_vocab_and_preserves_special_tokens(tmp_path):
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    input_path = tmp_path / "text.txt"
    output_path = tmp_path / "ids.npy"
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)
    with merges_path.open("wb") as f:
        pickle.dump([], f)
    input_path.write_text("hi<|endoftext|>ok\n", encoding="utf-8")

    count = tokenize_file(
        input_path,
        vocab_path,
        merges_path,
        output_path,
        special_tokens=["<|endoftext|>"],
    )

    ids = np.load(output_path)
    assert count == 6
    assert ids.tolist() == [104, 105, 256, 111, 107, 10]


def test_parallel_tokenize_file_matches_single_worker_across_chunks(tmp_path):
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    input_path = tmp_path / "text.txt"
    single_output = tmp_path / "single.npy"
    parallel_output = tmp_path / "parallel.npy"
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    with vocab_path.open("wb") as file:
        pickle.dump(vocab, file)
    with merges_path.open("wb") as file:
        pickle.dump([], file)
    input_path.write_text("hello\nworld\n<|endoftext|>\n", encoding="utf-8")

    tokenize_file(input_path, vocab_path, merges_path, single_output, ["<|endoftext|>"])
    tokenize_file(
        input_path,
        vocab_path,
        merges_path,
        parallel_output,
        ["<|endoftext|>"],
        workers=2,
        chunk_chars=4,
    )

    assert np.array_equal(np.load(single_output), np.load(parallel_output))
