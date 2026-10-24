"""New zero-initialized adapter; the visual backbone belongs to the host model."""
import torch
from torch import nn


def make_context_adapter(context_dim=768, hidden_dim=256, fused_dim=2048):
    if min(context_dim, hidden_dim, fused_dim) < 1:
        raise ValueError("Adapter dimensions must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3501)
        adapter = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim),
                                nn.LeakyReLU(), nn.Linear(hidden_dim, fused_dim, bias=False))
        nn.init.zeros_(adapter[-1].weight)
    return adapter


def context_residual(shared_encoder, adapter, mosaic, epoch=0, step=0):
    """One extra shared-encoder call, with the historical isolated RNG stream."""
    if mosaic.ndim != 4 or tuple(mosaic.shape[1:]) != (3, 224, 224):
        raise ValueError("Expected mosaics [B,3,224,224]")
    devices = [mosaic.device.index or 0] if mosaic.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(3501001 + int(epoch) * 100000 + int(step))
        context = shared_encoder(mosaic)
    if not isinstance(context, torch.Tensor) or context.ndim != 2 or context.shape[-1] != 768:
        raise ValueError("Shared visual encoder must return [B,768], not logits/tokens")
    return adapter(context)
