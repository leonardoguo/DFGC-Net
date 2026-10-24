# Downloaded weights

Place the official Swin-T ImageNet-1K 224 model here as
`swin_tiny_patch4_window7_224.pth`; it must contain a `model` state dict.

Official model hub:
https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md

Direct official asset:
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

No weight is included or downloaded automatically. An alternate local path can
be passed with `train.py --swin PATH` / `evaluate.py --swin PATH`.
Generated T-Net weights are saved in DATA_ROOT/tnet/, not this source folder.
