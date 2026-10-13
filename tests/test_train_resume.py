import torch
from torch import nn

from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.train import _restore_training_state


def test_restore_training_state_loads_model_optimizer_and_next_step(tmp_path):
    torch.manual_seed(7)
    saved_model = nn.Linear(3, 2)
    saved_optimizer = torch.optim.AdamW(saved_model.parameters(), lr=0.01)
    loss = saved_model(torch.ones(1, 3)).sum()
    loss.backward()
    saved_optimizer.step()

    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(saved_model, saved_optimizer, 10_000, checkpoint_path)

    resumed_model = nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)

    next_step = _restore_training_state(
        resumed_model, resumed_optimizer, checkpoint_path, map_location="cpu"
    )

    assert next_step == 10_001
    assert torch.equal(saved_model.weight, resumed_model.weight)
    assert torch.equal(saved_model.bias, resumed_model.bias)
    assert resumed_optimizer.state_dict()["state"].keys() == saved_optimizer.state_dict()["state"].keys()
