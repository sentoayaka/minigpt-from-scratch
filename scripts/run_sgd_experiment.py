import torch

from cs336_basics.optim import SGD

for lr in [1e1, 1e2, 1e3]:
    torch.manual_seed(0)

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    print(f"\nlr = {lr}")

    for t in range(10):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(t, loss.item())
        loss.backward()
        opt.step()