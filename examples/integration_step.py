"""Executable integration functions, not a replacement DA-Net trainer.

The caller supplies its own host model, dataloader, loss and optimizer.
See docs/INTEGRATION_zh.md for the batch fields and setup order.
"""


def training_step(host, extension, batch, epoch, step):
    metadata, pcs, pics, mos = batch
    with extension.batch(metadata["global_mosaic"], pcs.shape[0], pcs.shape[1],
                         epoch=epoch, step=step) as context:
        # This call is the user's separately obtained base implementation.
        prediction, base_losses = host(pcs, pics, mos)
        auxiliary, diagnostics = context.auxiliary_loss(metadata)
        loss = base_losses["loss"] + auxiliary
    return prediction, loss, diagnostics


def evaluation_step(host, extension, batch, epoch=0, step=0):
    metadata, pcs, pics, mos = batch
    with extension.batch(metadata["global_mosaic"], pcs.shape[0], pcs.shape[1],
                         epoch=epoch, step=step):
        prediction, base_losses = host(pcs, pics, mos)
    return prediction, base_losses
