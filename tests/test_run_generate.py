import os
import subprocess
import sys
from pathlib import Path

import torch

from scripts.run_generate import build_parser, default_device


def test_generate_defaults_load_the_owt_32k_checkpoint():
    args = build_parser().parse_args(["--prompt", "A short prompt"])

    assert args.prompt == "A short prompt"
    assert args.vocab_path == "outputs/owt_bpe_32k.vocab.pkl"
    assert args.merges_path == "outputs/owt_bpe_32k.merges.pkl"
    assert args.checkpoint_path == "outputs/checkpoints/owt_32k_latest.pt"
    assert args.vocab_size == 32_000
    assert args.context_length == 256
    assert args.d_model == 512
    assert args.num_layers == 4
    assert args.num_heads == 16
    assert args.d_ff == 1344


def test_default_device_prefers_gpu_four_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    assert default_device() == "cuda:4"


def test_default_device_falls_back_for_fewer_gpus_or_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert default_device() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert default_device() == "cpu"


def test_script_help_works_without_pythonpath_from_outside_project(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_generate.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Generate a continuation using the OWT 32k A1 checkpoint." in result.stdout
