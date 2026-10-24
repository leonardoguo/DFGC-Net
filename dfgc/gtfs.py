"""New geometry/texture supervision; no base encoder or DA-Net loss code."""
import torch
from torch import nn
from torch.nn import functional as F


def make_factor_heads(feature_dim=1024, hidden_dim=256):
    """Return independent heads in the historical geometry-then-texture order."""
    if min(feature_dim, hidden_dim) < 1:
        raise ValueError("Head dimensions must be positive")
    def head():
        return nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
                             nn.LeakyReLU(), nn.Linear(hidden_dim, 1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2001)
        return head(), head()


def masked_factor_loss(prediction, label, mask, beta=0.1):
    """Zero masks exclude a factor, rather than supervising it as severity zero."""
    if beta <= 0:
        raise ValueError("Smooth L1 beta must be positive")
    label = torch.as_tensor(label, device=prediction.device, dtype=prediction.dtype)
    mask = torch.as_tensor(mask, device=prediction.device, dtype=prediction.dtype)
    if label.numel() != prediction.numel() or mask.numel() != prediction.numel():
        raise ValueError("One severity and validity mask are required per prediction")
    label, mask = label.reshape_as(prediction), mask.reshape_as(prediction)
    if not torch.isfinite(label).all() or not torch.isfinite(mask).all():
        raise ValueError("Non-finite factor label/mask")
    if ((label < 0) | (label > 1)).any() or ((mask != 0) & (mask != 1)).any():
        raise ValueError("Severity must be in [0,1]; masks must be binary")
    error = F.smooth_l1_loss(prediction, label, beta=beta, reduction="none")
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def factor_losses(geometry_head, texture_head, geometry, texture, labels, beta=0.1):
    """Inputs [B,K,1024]; average within each cloud, not across its batch."""
    if geometry.ndim != 3 or texture.shape != geometry.shape:
        raise ValueError("Expected matching [B,K,C] modality features")
    geometry_prediction = torch.sigmoid(geometry_head(geometry.mean(dim=1)))
    texture_prediction = torch.sigmoid(texture_head(texture.mean(dim=1)))
    return {
        "geometry_loss": masked_factor_loss(geometry_prediction, labels["geometry_severity"],
                                           labels["geometry_severity_mask"], beta),
        "texture_loss": masked_factor_loss(texture_prediction, labels["texture_severity"],
                                          labels["texture_severity_mask"], beta),
        "geometry_prediction": geometry_prediction,
        "texture_prediction": texture_prediction,
    }
