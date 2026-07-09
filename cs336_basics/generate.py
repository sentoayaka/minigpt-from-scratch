import torch

from cs336_basics.model import softmax

@torch.no_grad()
def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    endoftext_token: str = "<|endoftext|>",
    device: str | torch.device = "cpu",
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    endoftext_token_id = tokenizer.token_to_id[endoftext_token.encode("utf-8")]

    for _ in range(max_new_tokens):
        context_length = getattr(model, "context_length", token_ids.size(1))
        logits = model(token_ids[:, -context_length:])

        next_token_logits = logits[0, -1, :]

        if temperature == 0.0:
            next_token_id = torch.argmax(next_token_logits).item()
        else:
            next_token_logits /= temperature
            probs = softmax(next_token_logits, dim=-1)

            if top_p is not None:
                probs = top_p_filtering(probs, top_p)

            next_token_id = torch.multinomial(probs, num_samples=1).item()

        token_ids = torch.cat(
            [
                token_ids, 
                torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            ], 
            dim=1
        )
        
        if next_token_id == endoftext_token_id:
            break

    return tokenizer.decode(token_ids[0].tolist())


def top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0

    return probs