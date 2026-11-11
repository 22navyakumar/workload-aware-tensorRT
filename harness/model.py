"""Load the shared model and build fixed, reproducible inputs."""
import torch
from transformers import AutoModel


def load_model(name, dtype=torch.float16, device="cuda"):
    model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return model


def make_inputs(batch, seq_len, vocab_size=30522, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g).to(device)
    mask = torch.ones_like(ids)
    return ids, mask
