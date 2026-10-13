import math

import torch
from torch import nn

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits = logits - max_logits
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    loss = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return loss.mean()