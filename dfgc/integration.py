"""Opt-in hooks around a separately obtained host; no host forward is copied."""
from contextlib import contextmanager
import torch
from torch import nn

from .gtfs import make_factor_heads, factor_losses
from .lgqc import make_context_adapter, context_residual


class DFGCExtensions:
    """Install before building the optimizer, wrapping with DDP, or loading weights.

    Required host endpoints: pc_linear/pic_linear -> [B*K,1024],
    pic_backbone -> [B,768], and fc input -> [B,2048].
    The host retains its forward, VQ/FDM/FMM, scoring head and base losses.
    """
    def __init__(self, host, geometry_weight=0.02, texture_weight=0.02, beta=0.1,
                 hidden_dim=256, preserve_cuda_rng=False):
        if min(geometry_weight, texture_weight) < 0 or beta <= 0:
            raise ValueError("Invalid factor-loss weights/beta")
        for name in ("pc_linear", "pic_linear", "pic_backbone", "fc"):
            if not isinstance(getattr(host, name, None), nn.Module):
                raise TypeError(f"Host must expose nn.Module {name!r}")
        new_names = ("gtfs_geometry_head", "gtfs_texture_head", "global_context_adapter")
        if any(hasattr(host, name) for name in new_names):
            raise ValueError("Extensions already installed or host uses conflicting names")
        states = torch.cuda.get_rng_state_all() if preserve_cuda_rng and torch.cuda.is_available() else None
        try:
            geometry, texture = make_factor_heads(1024, hidden_dim)
            adapter = make_context_adapter(768, hidden_dim, 2048)
        finally:
            if states is not None:
                torch.cuda.set_rng_state_all(states)
        # Keep checkpoint keys on the host, rather than prefixing them with base.*.
        reference = next(host.pc_linear.parameters())
        for name, module in zip(new_names, (geometry, texture, adapter)):
            module.to(device=reference.device, dtype=reference.dtype)
            module.train(host.training)
            setattr(host, name, module)
        self.host = host
        self.geometry_weight, self.texture_weight, self.beta = geometry_weight, texture_weight, beta
        self._active = False

    @contextmanager
    def batch(self, mosaic, batch_size, patch_count, *, epoch=0, step=0):
        """Enclose exactly one ordinary host forward; always remove temporary hooks."""
        if self._active:
            raise RuntimeError("Nested/concurrent batch contexts are unsupported")
        if batch_size < 1 or patch_count < 1 or tuple(mosaic.shape) != (batch_size, 3, 224, 224):
            raise ValueError("Mosaic batch does not match batch_size/patch_count")
        context = _Batch(self, mosaic, batch_size, patch_count, epoch, step)
        handles = []
        self._active = True
        try:
            handles.append(self.host.pc_linear.register_forward_hook(context.capture_geometry))
            handles.append(self.host.pic_linear.register_forward_hook(context.capture_texture))
            handles.append(self.host.fc.register_forward_pre_hook(context.add_context))
            yield context
            context.validate()
        finally:
            for handle in handles:
                handle.remove()
            context.features.clear()
            context.mosaic = None
            self._active = False


class _Batch:
    def __init__(self, extension, mosaic, batch_size, patch_count, epoch, step):
        self.extension, self.mosaic = extension, mosaic
        self.batch_size, self.patch_count = batch_size, patch_count
        self.epoch, self.step = epoch, step
        self.features, self.score_calls = {}, 0

    def _capture(self, key, value):
        expected = (self.batch_size * self.patch_count, 1024)
        if key in self.features or not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"Host {key} projection must run once and return {expected}")
        self.features[key] = value.reshape(self.batch_size, self.patch_count, 1024)

    def capture_geometry(self, module, args, output):
        self._capture("geometry", output)

    def capture_texture(self, module, args, output):
        self._capture("texture", output)

    def add_context(self, module, args):
        self.score_calls += 1
        if self.score_calls != 1 or len(args) != 1 or tuple(args[0].shape) != (self.batch_size, 2048):
            raise ValueError("Host scoring head must be called once with [B,2048]")
        host = self.extension.host
        delta = context_residual(host.pic_backbone, host.global_context_adapter,
                                 self.mosaic, self.epoch, self.step)
        return (args[0] + delta,)

    def validate(self):
        if self.score_calls != 1 or set(self.features) != {"geometry", "texture"}:
            raise RuntimeError("Host did not execute the expected feature/scoring endpoints")

    def auxiliary_loss(self, labels):
        """Call inside the context, after host forward; do not add again at validation."""
        self.validate()
        ext, host = self.extension, self.extension.host
        values = factor_losses(host.gtfs_geometry_head, host.gtfs_texture_head,
                               self.features["geometry"], self.features["texture"], labels, ext.beta)
        total = ext.geometry_weight * values["geometry_loss"] + ext.texture_weight * values["texture_loss"]
        return total, values
