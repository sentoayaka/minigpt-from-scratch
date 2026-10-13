import numpy as np
import torch

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = len(dataset) - context_length
    indices = np.random.randint(0, num_samples, size=batch_size)

    inputs = np.array([dataset[i:i + context_length] for i in indices])
    targets = np.array([dataset[i + 1:i + context_length + 1] for i in indices])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets