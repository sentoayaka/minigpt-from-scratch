import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


DATA_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def write_status(path, phase, **fields):
    payload = {
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def run_logged(command, cwd, env, log_path, status_path, phase):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        write_status(status_path, phase, pid=process.pid, command=command)
        return process.wait()


def ensure_download(url, destination, status_path, split):
    if destination.is_file() and destination.stat().st_size > 0:
        return
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(10):
        offset = partial.stat().st_size if partial.exists() else 0
        write_status(
            status_path,
            f"downloading_{split}",
            downloaded_bytes=offset,
            attempt=attempt + 1,
        )
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = offset > 0 and response.status == 206
                mode = "ab" if resumed else "wb"
                content_range = response.headers.get("Content-Range", "")
                if resumed and "/" in content_range:
                    expected_size = int(content_range.rsplit("/", 1)[1])
                else:
                    content_length = response.headers.get("Content-Length")
                    expected_size = int(content_length) if content_length else None
                bytes_since_status = 0
                downloaded_bytes = offset if resumed else 0
                with partial.open(mode) as output:
                    while chunk := response.read(8 * 1024 * 1024):
                        output.write(chunk)
                        bytes_since_status += len(chunk)
                        downloaded_bytes += len(chunk)
                        if bytes_since_status >= 256 * 1024 * 1024:
                            write_status(
                                status_path,
                                f"downloading_{split}",
                                downloaded_bytes=downloaded_bytes,
                                expected_bytes=expected_size,
                                attempt=attempt + 1,
                            )
                            bytes_since_status = 0

            downloaded = partial.stat().st_size
            if expected_size is not None and downloaded != expected_size:
                if downloaded < expected_size:
                    write_status(
                        status_path,
                        f"downloading_{split}",
                        downloaded_bytes=downloaded,
                        expected_bytes=expected_size,
                        attempt=attempt + 1,
                    )
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"downloaded {downloaded} bytes, expected {expected_size}"
                )
            if downloaded == 0:
                raise RuntimeError("download returned an empty file")
            print(f"downloaded TinyStories {split}: {downloaded:,} bytes", flush=True)
            write_status(
                status_path,
                f"downloaded_{split}",
                downloaded_bytes=downloaded,
                expected_bytes=expected_size,
            )
            break
        except Exception as error:
            if attempt == 9:
                raise RuntimeError(
                    f"TinyStories {split} download failed after 10 attempts"
                ) from error
            print(
                f"TinyStories {split} download attempt {attempt + 1} failed: {error}; retrying",
                flush=True,
            )
            time.sleep(2)

    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"TinyStories {split} download failed")
    os.replace(partial, destination)


def main():
    parser = argparse.ArgumentParser(
        description="Download TinyStories, train its A1 BPE tokenizer, and run 4-GPU training."
    )
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--physical-gpus", default="4,5,6,7")
    args = parser.parse_args()

    project = args.project.resolve()
    data_dir = project / "data"
    output_dir = project / "outputs"
    logs_dir = output_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    status_path = output_dir / "tinystories_a1_train.status.json"
    train_text = data_dir / "TinyStoriesV2-GPT4-train.txt"
    valid_text = data_dir / "TinyStoriesV2-GPT4-valid.txt"
    vocab_path = output_dir / "tinystories_bpe_10k.vocab.pkl"
    merges_path = output_dir / "tinystories_bpe_10k.merges.pkl"
    train_ids = output_dir / "tinystories_train_ids.npy"
    valid_ids = output_dir / "tinystories_valid_ids.npy"
    checkpoint_path = output_dir / "checkpoints" / "tinystories_10k_latest.pt"
    metrics_path = logs_dir / "tinystories_10k_train.csv"
    python = sys.executable

    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(project),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )

    ensure_download(DATA_URLS["train"], train_text, status_path, "train")
    ensure_download(DATA_URLS["valid"], valid_text, status_path, "valid")

    if not (vocab_path.is_file() and merges_path.is_file()):
        bpe_command = [
            python,
            "scripts/run_bpe_experiment.py",
            "--input-path",
            str(train_text),
            "--vocab-size",
            "10000",
            "--out-prefix",
            str(output_dir / "tinystories_bpe_10k"),
        ]
        result = run_logged(
            bpe_command,
            project,
            env,
            logs_dir / "tinystories_bpe_10k.log",
            status_path,
            "training_tinystories_bpe_10k",
        )
        if result != 0 or not (vocab_path.is_file() and merges_path.is_file()):
            raise RuntimeError("TinyStories BPE failed; see outputs/logs/tinystories_bpe_10k.log")

    tokenizer_script = project / "scripts" / "tokenize_corpus.py"
    for split, source, destination in (
        ("train", train_text, train_ids),
        ("valid", valid_text, valid_ids),
    ):
        if destination.exists():
            continue
        command = [
            python,
            str(tokenizer_script),
            "--input",
            str(source),
            "--vocab",
            str(vocab_path),
            "--merges",
            str(merges_path),
            "--output",
            str(destination),
            "--special-token",
            "<|endoftext|>",
        ]
        result = run_logged(
            command,
            project,
            env,
            logs_dir / f"tinystories_{split}_tokenize.log",
            status_path,
            f"tokenizing_{split}",
        )
        if result != 0 or not destination.exists():
            raise RuntimeError(f"TinyStories {split} tokenization failed; see outputs/logs/tinystories_{split}_tokenize.log")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=args.physical_gpus, NCCL_DEBUG="WARN")
    train_command = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "--module",
        "cs336_basics.train",
        "--train_data",
        str(train_ids),
        "--valid_data",
        str(valid_ids),
        "--checkpoint_path",
        str(checkpoint_path),
        "--log_path",
        str(metrics_path),
        "--device",
        "cuda:0",
        "--vocab_size",
        "10000",
        "--context_length",
        "256",
        "--d_model",
        "512",
        "--num_layers",
        "4",
        "--num_heads",
        "16",
        "--d_ff",
        "1344",
        "--batch_size",
        "16",
        "--num_steps",
        "20000",
        "--max_lr",
        "3e-4",
        "--min_lr",
        "3e-5",
        "--warmup_iters",
        "100",
        "--cosine_cycle_iters",
        "20000",
        "--eval_every",
        "1000",
        "--eval_iters",
        "10",
        "--save_every",
        "5000",
        "--log_every",
        "10",
        "--seed",
        "0",
    ]
    result = run_logged(
        train_command,
        project,
        env,
        logs_dir / "tinystories_4gpu_train.log",
        status_path,
        "training_tinystories_4gpu",
    )
    if result != 0:
        write_status(status_path, "failed", returncode=result)
        raise RuntimeError("4-GPU TinyStories training failed; see outputs/logs/tinystories_4gpu_train.log")

    write_status(
        status_path,
        "complete",
        checkpoint=str(checkpoint_path),
        metrics=str(metrics_path),
        num_steps=20000,
        global_batch_size=64,
        tokens_processed=327_680_000,
    )


if __name__ == "__main__":
    main()
