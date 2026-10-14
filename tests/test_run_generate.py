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
