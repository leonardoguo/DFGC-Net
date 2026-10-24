# DFGC-Net

Distortion Factor Supervision and Global Context Fusion for No-Reference Point
Cloud Quality Assessment.

DFGC-Net combines geometry–texture distortion factor supervision (GTFS) and
global quality context (LGQC) for no-reference point cloud quality assessment.
This repository supports training and evaluation on WPC and SJTU-PCQA.

**Download DA-Net separately before running the code.** The entrypoints load its
base components from the directory supplied with `--upstream` and install our
extensions automatically. No manual modification of DA-Net is required.

Start with the complete reproduction guide:
[English](docs/READER_GUIDE.md) | [简体中文](docs/READER_GUIDE_zh.md).
They list download links, exact folder placement, every command and output.
[Protocol](docs/REPRODUCTION.md) · [advanced integration (中文)](docs/INTEGRATION_zh.md)
· [scope/attribution](NOTICE.md)

## Layout

```text
workspace/
├── DFGC-Net/                       # this repository
│   ├── prepare.py                 # metadata / patches / views / T-Net / PNG / check
│   ├── train.py                   # all folds; automatic last-checkpoint resume
│   ├── evaluate.py                # one saved fold checkpoint
│   ├── dfgc/                      # additions and orchestration, not base code
│   ├── configs/                   # WPC/SJTU protocol and ordered split recipes
│   ├── pretrained/                # user-downloaded Swin weight
│   └── outputs/                   # generated training results; Git-ignored
├── DA-Net/                        # obtain separately; contains main.py/model/data
└── data/
    ├── wpc/raw/                   # original distorted WPC PLYs
    └── sjtu/
        ├── distortion/            # original distorted SJTU PLYs
        └── Final_MOS.mat          # official 42x9 MOS matrix
```

Do not place third-party source inside DFGC-Net or commit data/weights to Git.
Paths below are examples relative to the DFGC-Net root.

## Environment

Reference: Python3.9, PyTorch2.2.0, CUDA11.8, Lightning2.2.1.
Install a matching PyTorch first, then reader dependencies and PyTorch3D as
explained in the guide. PyTorch3D must be importable by the supplied base code;
having image caches does not remove a top-level import in an upstream revision.

```bash
pip install -r requirements-reader.txt
python -c "import torch; from pytorch3d import ops; print(torch.__version__, torch.cuda.is_available())"
```

Place `swin_tiny_patch4_window7_224.pth` in `pretrained/`, or pass `--swin PATH`.
Use the original Swin-T ImageNet-1K 224 model, not Swin-V2/22K weights.
No third-party download/install is performed automatically.

## From raw data (SJTU example)

After downloading DA-Net and the dataset, run in order:

```bash
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage metadata
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage patches
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage views
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage tnet
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage png
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage check
python train.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --folds 1 --smoke
python train.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu
```

WPC uses `--dataset wpc --data-root ../data/wpc`. Its metadata stage reads the
user-downloaded DA-Net fold1 tables to obtain WPC MOS/generation parameters,
then builds our fixed five-fold split order. SJTU reads official Final_MOS.mat.
A normalized external metadata CSV/XLSX may be supplied with `--metadata PATH`.

T-Net orchestration uses the external architecture/alignment loss with cached
K50 normals, 28 train/4 fixed monitor patches per cloud, no MOS, at most100
epochs and patience10. PNG generation calls the external projection functions
with the best exported T-Net, not last.ckpt. This all-unlabelled preprocessing
includes quality held-out content. Global six-view rendering is our implementation.
Raw WPC regeneration does not guarantee historical PNGs/results: use the
author's historical WPC caches when verifying those results.

## With compatible historical local caches

Put 32-patch PLYs in `DATA_ROOT/patches/` and local PNGs in
`DATA_ROOT/projections/`. Then run metadata, views, check and train; skip
patches/tnet/png. Never run generation over an imported cache without provenance.
All caches must cover every distorted sample and use the documented naming.

## Training, resume and results

Defaults: WPC5/SJTU9 folds,100 epochs/fold, seed42, batch4,
8 training patches and32 held-out patches. Both additions are always installed.
`--folds 1 4` selects folds; `--workers 0` disables multiprocessing;
`--dry-run` checks inputs without training. The ordinary runner uses a CUDA GPU;
`--device cpu` is available for checks, not a speed recommendation.

`--smoke` uses four batches per loop by default and verifies actual optimizer
updates plus finite model/optimizer states. Set `--smoke-batches N` to adjust it.
This avoids mistaking an AMP-skipped first update for a completed training step.

Results: `outputs/DATASET/fold_N/version_0/` with config/source hashes,
top3+last checkpoints, per-epoch prediction CSVs, TensorBoard and completed.json.
`outputs/DATASET/summary.csv` reports fold results and an explicitly labelled
all-fold or available-fold mean. Re-run the same command to resume last.ckpt;
completed folds are skipped. Changes to config/base source/extension source are
rejected in an existing run: use a fresh `--output PATH`. Resume does not
guarantee bitwise-identical persistent-worker augmentation streams.

## Evaluate a saved checkpoint

```bash
python evaluate.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --fold 1 --checkpoint PATH_TO_BEST.ckpt --output ../results/sjtu_fold1.csv
```

Use the best checkpoint recorded by that fold's completed.json, not a guessed
epoch. Evaluation strictly loads the combined state and activates LGQC.
SROCC/KROCC use raw predictions; PLCC/RMSE use the supplied base implementation's
fitting. The original reference used unconstrained five-parameter fitting.
Selection is best held-out-fold SROCC each epoch, with no separate inner validation.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

The test suite covers factor supervision, residual initialization, gradient
propagation, occupancy sampling, label mappings and metadata/split conversion.
See [test coverage and validation scope](docs/VERIFICATION.md) for details.

Full-backbone GPU smoke tests passed for WPC and SJTU fold1 on an RTX4060 Laptop
GPU with FP16 AMP, using four training and four validation batches per dataset.
These functional checks do not constitute a new full cross-validation experiment.

## Acknowledgements

This work builds on [DA-Net](https://github.com/Rphone/DA-Net). Compact multi-view
mosaic organization is inspired by GMS-3DQA. The visual encoder uses
[Swin Transformer](https://github.com/microsoft/Swin-Transformer); local projection
depends on [PyTorch3D](https://github.com/facebookresearch/pytorch3d).
Please acknowledge the corresponding methods and datasets when using this work.

## External resources

Base implementations, MOS, datasets, cached images and weights are obtained
separately through the links in the reader guide. This release covers the combined
method on WPC and SJTU-PCQA, not LS-PCQA or separate ablation runners.
External resources retain their original terms; see [NOTICE.md](NOTICE.md).

## License

The code contributed in this repository is released under the [MIT License](LICENSE).
You may use, modify and redistribute it, including commercially, provided that
the copyright and license notices are preserved. The software is provided without
warranty.

This license does not cover the separately obtained DA-Net implementation,
datasets, caches, pretrained checkpoints or other external resources; their own
terms apply. See [NOTICE.md](NOTICE.md) for the distribution scope.
