import numpy as np
import pickle
import pytest

from scripts import run_tinystories_training_pipeline as pipeline
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


def test_tinystories_download_resumes_partial_file(tmp_path, monkeypatch):
    class Response:
        status = 206
        headers = {"Content-Range": "bytes 2-5/6", "Content-Length": "4"}

        def __init__(self):
            self.chunks = [b"cde", b"f", b""]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _size):
            return self.chunks.pop(0)

    response = Response()
    requested_headers = {}

    def fake_urlopen(request, timeout):
        requested_headers.update(request.header_items())
        assert timeout == 60
        return response

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "train.txt"
    destination.with_name("train.txt.part").write_bytes(b"ab")
    status_path = tmp_path / "status.json"

    pipeline.ensure_download("https://example.test/train.txt", destination, status_path, "train")

    assert destination.read_bytes() == b"abcdef"
    assert not destination.with_name("train.txt.part").exists()
    assert requested_headers["Range"] == "bytes=2-"
