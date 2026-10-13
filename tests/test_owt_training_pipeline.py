import pickle

import numpy as np
import pytest

from scripts.run_owt_training_pipeline import (
    build_training_command,
    token_count,
    validate_bpe_artifacts,
)


def test_validate_bpe_artifacts_accepts_expected_sizes(tmp_path):
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    with vocab_path.open("wb") as file:
        pickle.dump({0: b"a", 1: b"b"}, file)
    with merges_path.open("wb") as file:
        pickle.dump([(b"a", b"b")], file)

    assert validate_bpe_artifacts(
        vocab_path, merges_path, expected_vocab_size=2, expected_merge_count=1
    ) == (2, 1)


def test_validate_bpe_artifacts_rejects_wrong_vocab_size(tmp_path):
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    with vocab_path.open("wb") as file:
        pickle.dump({0: b"a"}, file)
    with merges_path.open("wb") as file:
        pickle.dump([], file)

    with pytest.raises(ValueError, match="vocab size"):
        validate_bpe_artifacts(
            vocab_path, merges_path, expected_vocab_size=2, expected_merge_count=1
        )


def test_build_training_command_uses_a1_owt_four_gpu_defaults():
    command = build_training_command(
        python="python",
        train_ids="outputs/owt_train_ids.npy",
        valid_ids="outputs/owt_valid_ids.npy",
        checkpoint="outputs/checkpoints/owt_32k_latest.pt",
        metrics="outputs/logs/owt_32k_train.csv",
        nproc_per_node=4,
    )

    assert command[:6] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "--module",
    ]
    assert command[6:8] == ["cs336_basics.train", "--train_data"]
    options = dict(zip(command[7::2], command[8::2]))
    assert options["--vocab_size"] == "32000"
    assert options["--batch_size"] == "16"
    assert options["--num_steps"] == "20000"
    assert options["--eval_every"] == "1000"
    assert options["--save_every"] == "5000"


def test_token_count_accepts_uint16_vector_with_training_context(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.zeros(300, dtype=np.uint16))

    assert token_count(path) == 300


def test_token_count_rejects_wrong_dtype(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.zeros(300, dtype=np.int64))

    with pytest.raises(ValueError, match="uint16"):
        token_count(path)

