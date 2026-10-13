import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np


OWT_VOCAB_SIZE = 32_000
OWT_MERGE_COUNT = 31_743
CONTEXT_LENGTH = 256
PER_RANK_BATCH_SIZE = 16
NUM_STEPS = 20_000
EVAL_EVERY = 1_000
SAVE_EVERY = 5_000
TOKENIZER_WORKERS = 16


def validate_bpe_artifacts(
    vocab_path,
    merges_path,
    expected_vocab_size=OWT_VOCAB_SIZE,
    expected_merge_count=OWT_MERGE_COUNT,
):
    with Path(vocab_path).open("rb") as file:
        vocab = pickle.load(file)
    with Path(merges_path).open("rb") as file:
        merges = pickle.load(file)

    if len(vocab) != expected_vocab_size:
        raise ValueError(
            f"OWT BPE vocab size is {len(vocab)}, expected {expected_vocab_size}"
        )
    if len(merges) != expected_merge_count:
        raise ValueError(
            f"OWT BPE merge count is {len(merges)}, expected {expected_merge_count}"
        )
    return len(vocab), len(merges)


def token_count(path):
    tokens = np.load(path, mmap_mode="r")
    if tokens.dtype != np.uint16:
        raise ValueError(f"token file must use uint16, got {tokens.dtype}: {path}")
    if tokens.ndim != 1 or tokens.size <= CONTEXT_LENGTH:
        raise ValueError(
            f"token file must be a 1-D sequence longer than {CONTEXT_LENGTH}: {path}"
        )
    return int(tokens.size)


def build_training_command(
    python, train_ids, valid_ids, checkpoint, metrics, nproc_per_node=4, resume_from=None
):
    command = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "--module",
        "cs336_basics.train",
        "--train_data",
        str(train_ids),
        "--valid_data",
        str(valid_ids),
        "--checkpoint_path",
        str(checkpoint),
        "--log_path",
        str(metrics),
        "--device",
        "cuda:0",
        "--vocab_size",
        str(OWT_VOCAB_SIZE),
        "--context_length",
        str(CONTEXT_LENGTH),
        "--d_model",
        "512",
        "--num_layers",
        "4",
        "--num_heads",
        "16",
        "--d_ff",
        "1344",
        "--batch_size",
        str(PER_RANK_BATCH_SIZE),
        "--num_steps",
        str(NUM_STEPS),
        "--max_lr",
        "3e-4",
        "--min_lr",
        "3e-5",
        "--warmup_iters",
        "100",
        "--cosine_cycle_iters",
        str(NUM_STEPS),
        "--eval_every",
        str(EVAL_EVERY),
        "--eval_iters",
        "10",
        "--save_every",
        str(SAVE_EVERY),
        "--log_every",
        "10",
        "--seed",
        "0",
    ]
    if resume_from is not None:
        command.extend(["--resume_from", str(resume_from)])
    return command


def build_tokenize_command(
    python, tokenizer_script, source, vocab, merges, destination
):
    return [
        python,
        str(tokenizer_script),
        "--input",
        str(source),
        "--vocab",
        str(vocab),
        "--merges",
        str(merges),
        "--output",
        str(destination),
        "--special-token",
        "<|endoftext|>",
        "--workers",
        str(TOKENIZER_WORKERS),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize OWT with the existing 32k BPE and run A1 training."
    )
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--physical-gpus", default="4,5,6,7")
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    data_dir = project / "data"
    output_dir = project / "outputs"
    logs_dir = output_dir / "logs"
    status_path = output_dir / "owt_a1_train.status.json"
    train_text = data_dir / "owt_train.txt"
    valid_text = data_dir / "owt_valid.txt"
    vocab_path = output_dir / "owt_bpe_32k.vocab.pkl"
    merges_path = output_dir / "owt_bpe_32k.merges.pkl"
    train_ids = output_dir / "owt_train_ids.npy"
    valid_ids = output_dir / "owt_valid_ids.npy"
    checkpoint_path = output_dir / "checkpoints" / "owt_32k_latest.pt"
    resume_from = args.resume_from
    if resume_from is not None:
        if not resume_from.is_absolute():
            resume_from = project / resume_from
        resume_from = resume_from.resolve()
    metrics_path = logs_dir / (
        "owt_32k_resume.csv" if resume_from else "owt_32k_train.csv"
    )
    training_log_path = logs_dir / (
        "owt_4gpu_train_resume.log" if resume_from else "owt_4gpu_train.log"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(project),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )

    def set_status(phase, **details):
        temporary = status_path.with_name(status_path.name + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "phase": phase,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    **details,
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, status_path)

    def run_logged(command, log_path, phase):
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                cwd=project,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            set_status(phase, pid=process.pid, command=command)
            return process.wait()

    try:
        for required in (train_text, valid_text, vocab_path, merges_path):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(
                    f"required OWT input or completed BPE artifact is missing: {required}"
                )

        vocab_count, merge_count = validate_bpe_artifacts(vocab_path, merges_path)
        if resume_from is not None and not resume_from.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_from}")
        set_status(
            "bpe_ready",
            vocab=str(vocab_path),
            merges=str(merges_path),
            vocab_size=vocab_count,
            merge_count=merge_count,
        )

        tokenizer_script = project / "scripts" / "tokenize_corpus.py"
        for split, source, destination in (
            ("train", train_text, train_ids),
            ("valid", valid_text, valid_ids),
        ):
            if destination.exists():
                token_count(destination)
                continue
            command = build_tokenize_command(
                sys.executable,
                tokenizer_script,
                source,
                vocab_path,
                merges_path,
                destination,
            )
            result = run_logged(
                command,
                logs_dir / f"owt_{split}_tokenize.log",
                f"tokenizing_{split}",
            )
            if result != 0:
                raise RuntimeError(
                    f"OWT {split} tokenization failed; see logs/owt_{split}_tokenize.log"
                )
            token_count(destination)

        train_token_count = token_count(train_ids)
        valid_token_count = token_count(valid_ids)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        gpu_ids = [gpu.strip() for gpu in args.physical_gpus.split(",") if gpu.strip()]
        if not gpu_ids:
            raise ValueError("--physical-gpus must contain at least one GPU ID")
        env.update(CUDA_VISIBLE_DEVICES=",".join(gpu_ids), NCCL_DEBUG="WARN")
        train_command = build_training_command(
            sys.executable,
            train_ids,
            valid_ids,
            checkpoint_path,
            metrics_path,
            nproc_per_node=len(gpu_ids),
            resume_from=resume_from,
        )
        result = run_logged(
            train_command,
            training_log_path,
            "training_owt",
        )
        if result != 0:
            raise RuntimeError(
                f"OWT training failed; see {training_log_path}"
            )

        set_status(
            "complete",
            checkpoint=str(checkpoint_path),
            metrics=str(metrics_path),
            train_tokens=train_token_count,
            valid_tokens=valid_token_count,
            num_steps=NUM_STEPS,
            world_size=len(gpu_ids),
            physical_gpus=gpu_ids,
            per_rank_batch_size=PER_RANK_BATCH_SIZE,
            global_batch_size=PER_RANK_BATCH_SIZE * len(gpu_ids),
            context_length=CONTEXT_LENGTH,
            **({"resumed_from": str(resume_from)} if resume_from else {}),
            tokens_processed=(
                NUM_STEPS * PER_RANK_BATCH_SIZE * len(gpu_ids) * CONTEXT_LENGTH
            ),
        )
    except Exception as error:
        set_status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
