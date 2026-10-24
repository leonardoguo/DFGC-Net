# Reproduction Guide: Download, Organize, and Run

**English** | [简体中文](READER_GUIDE_zh.md)

This guide covers the combined DFGC-Net model (GTFS + LGQC) on WPC and
SJTU-PCQA. No manual changes to DA-Net are required: our entrypoints import its
base components from the directory specified by `--upstream`.

Run all commands from the DFGC-Net repository root. Adjust the example paths to
your installation. Use a terminal with the required arguments rather than
opening a Python file without arguments.

## 1. Required downloads

| Resource | Download source | Location |
|---|---|---|
| DFGC-Net source | This repository | `workspace/DFGC-Net/` |
| DA-Net base implementation | [Official repository](https://github.com/Rphone/DA-Net): Code → Download ZIP, or Git | `workspace/DA-Net/`, directly containing `main.py`, `model/`, and `data/` |
| Swin-T ImageNet-1K pretrained checkpoint | [Microsoft model hub](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md): Swin-T, 224×224, ImageNet-1K | `DFGC-Net/pretrained/swin_tiny_patch4_window7_224.pth` |
| Raw distorted WPC point clouds | [Official WPC repository](https://github.com/qdushl/Waterloo-Point-Cloud-Database), using its README's dataset link | `workspace/data/wpc/raw/`; subdirectories are supported |
| Historical WPC local patches and PNGs | Download links in the [DA-Net README](https://github.com/Rphone/DA-Net#download-the-preprocessed-wpc-datasets-and-pretrained-checkpoints) | PLYs in `data/wpc/patches/`; PNGs in `data/wpc/projections/` |
| SJTU-PCQA point clouds and MOS | [Dataset authors' page](https://vision.nju.edu.cn/28/fd/c29466a469245/page.htm) | Distorted PLYs in `data/sjtu/distortion/`; `Final_MOS.mat` in `data/sjtu/` |

Official Swin checkpoint: [swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth).
Use Swin-V1 Tiny pretrained on ImageNet-1K, not Swin-V2 or ImageNet-22K weights.

WPC cache links listed by DA-Net:

- [Baidu Netdisk](https://pan.baidu.com/s/1qXzm8nF_H5ABftAJzsD1sw?pwd=wwd3), access code `wwd3`.
- [OneDrive](https://1drv.ms/f/c/6900cdc80191473b/IgBpMxxEKdAjT4V1KnnltpHQAZpYLgHz0ck6ZBPY04-dmPQ).

SJTU download links listed on the dataset authors' page:

- [Baidu Netdisk](https://pan.baidu.com/s/1x6ZqETv4b3O0a6LB0oV7jg), access code `rzgk`.
- [Google Drive](https://drive.google.com/file/d/13L1opI6dM5eiNkwy9pu0URfI09hyIV8a/view?usp=sharing).

These resources are maintained by third parties. If a link changes, consult the
corresponding official page and follow the resource owner's terms.

The supplied protocols use 740 distorted WPC samples and 378 distorted SJTU
samples. Keep reference point clouds separate from distorted inputs.
Retain SJTU's official `Final_MOS.mat`; do not treat a headerless `MOS.xlsx`
matrix as a table of named samples.

For WPC, the metadata stage reads the two fold-1 spreadsheets in the separately
downloaded DA-Net repository, then builds this project's fixed five-fold splits.
MOS tables are not bundled here. An externally obtained, normalized metadata
table can also be supplied with `--metadata PATH`.

Training from scratch requires the Swin pretrained checkpoint, not a trained
DA-Net quality checkpoint. T-Net is trained during input preparation unless
compatible local PNGs are already available.

## 2. Directory layout

```text
workspace/
├── DFGC-Net/
│   ├── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   └── pretrained/swin_tiny_patch4_window7_224.pth
├── DA-Net/
│   ├── main.py
│   ├── model/IQA_Model.py
│   ├── data/Datasets.py
│   └── configs/wpc/1/{train.xlsx,vail.xlsx}
└── data/
    ├── wpc/
    │   ├── raw/                    # raw distorted point clouds
    │   ├── patches/                # downloaded or generated 32-patch PLYs
    │   └── projections/            # corresponding local PNGs
    └── sjtu/
        ├── distortion/             # raw distorted point clouds
        └── Final_MOS.mat
```

A ZIP download may introduce an extra parent directory. `--upstream` must point
to the directory directly containing DA-Net's `main.py`, not the outer wrapper.
Keep external source outside DFGC-Net. Do not commit datasets, generated caches,
outputs, or weights to the source repository.

## 3. Environment setup

Use an isolated environment. The reference setup is Python 3.9, PyTorch 2.2.0,
CUDA 11.8, and PyTorch Lightning 2.2.1:

```bash
conda create -n dfgc python=3.9 -y
conda activate dfgc
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-reader.txt
```

Install a compatible PyTorch3D build separately using the
[official installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
On Windows, a prebuilt wheel must match Python, PyTorch, and CUDA; third-party
wheels are not included here. Installing an arbitrary wheel does not establish
binary compatibility. Some DA-Net revisions import PyTorch3D at module level,
so it must be importable even when using existing image caches.

Check the environment and external interfaces:

```bash
python -c "import torch; from pytorch3d import ops; print(torch.__version__, torch.cuda.is_available())"
python scripts/inspect_upstream.py --upstream ../DA-Net
python -B -m unittest discover -s tests -v
```

If extension installation reports conflicting names, supply an unextended DA-Net,
not a version already containing GTFS/LGQC experiments. Resolve missing-interface
errors by checking the base revision rather than bypassing validation.
See [test coverage and validation scope](VERIFICATION.md).

## 4. WPC: use the historical local caches

Place downloaded PLYs and PNGs directly in the destinations shown above. Expected
names are `patches/<name>_part0.ply` and `projections/<name>_part0_1.png`, through
`part31`. Do not leave an extra `pc/` or `proj/` directory nested inside them.
Raw distorted WPC point clouds are still needed for the global six-view inputs.

Run in order:

```bash
python prepare.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --stage metadata
python prepare.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --stage views
python prepare.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --stage check
python train.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --folds 1 --dry-run
python train.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --folds 1 --smoke
python train.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc
```

The last command trains folds 1–5 sequentially. `--dry-run` checks inputs without
training. `--smoke` runs four training and four validation batches by default,
and verifies that an optimizer update occurred. Adjust this with `--smoke-batches`.
This is not a performance experiment and does not produce formal trained results.
AMP may skip an initial update while adjusting its loss scale; a single batch
is therefore insufficient to verify optimizer updates reliably.

## 5. SJTU: prepare inputs from raw data and train

Place `distortion/` and `Final_MOS.mat` under `data/sjtu/`, then run:

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

The last command trains folds 1–9 sequentially. Preparation writes
`metadata.csv`, `splits/`, `patches/`, `global_views/`, `tnet_normals_k50/`,
`tnet/`, and `projections/` inside the dataset root. Patch generation runs
serially with fixed per-sample seeds and may take time.

T-Net uses geometry/normals without MOS. Each cloud contributes 28 training
patches and four fixed monitoring patches. This preprocessing includes unlabelled
geometry from the quality held-out contents. Training runs for at most 100 epochs,
with early stopping after 10 epochs without sufficient monitoring-loss improvement.

The T-Net stage writes `tnet/best.ckpt` and `tnet/last.ckpt`. PNG generation loads
`best.ckpt` automatically; `--tnet-checkpoint PATH` selects another suitable
checkpoint. Re-run the same T-Net command after an interruption. Changing the
settings is not a continuation of the same run.

Generated caches have provenance records. Preparation refuses to overwrite
unrecorded imported caches or mix changed recipes into an existing cache.

## 6. Generate WPC local inputs from raw data

Use a fresh WPC data root without imported local caches. Run all preparation
stages from Section 5 with `--dataset wpc` and the WPC data root.
Raw filenames ending in `_rounded` are recognized as aliases; original files
are not renamed.

The T-Net weights used to produce the historical WPC PNGs are unavailable.
Training another T-Net provides a complete workflow, but does not guarantee
pixel-identical PNGs or identical final metrics. Use Section 4's cache route to
check the historical results. Hardware, software versions, and regenerated inputs
can also affect training outcomes.

## 7. Training settings, resume, and outputs

Defaults are 100 epochs per fold, main seed 42, batch size four clouds, eight
patches per training cloud, and all 32 patches during validation. WPC MOS is
divided by 100; SJTU MOS is divided by 10 and its local XYZRGB is normalized by
the base loader. Dataset-specific factor mappings are selected automatically.
No manual assembly of the model or training loss is required.

Outputs are saved under `DFGC-Net/outputs/<dataset>/fold_N/version_0/`:

- `config.json`: settings and source/checkpoint/split hashes.
- `checkpoints/`: the top three SROCC checkpoints and `last.ckpt`.
- `predictions/epoch_XXX.csv`: sample names, predictions, and MOS for each epoch.
- `completed.json`: the selected epoch, checkpoint path, and metrics.
- TensorBoard event files: training logs.

`outputs/<dataset>/summary.csv` contains completed-fold results and their arithmetic
mean. An incomplete run is labelled `mean_available`; a complete run is labelled
`mean_all_folds`. An available-fold mean is not a full cross-validation result.
Reserve sufficient disk space for every fold's top-three and last checkpoints.

Re-run the same training command to restore `last.ckpt` after an interruption.
Completed folds are skipped. Configuration/source/base-revision changes require
a new `--output PATH`. Use `--folds 1 4` to select folds and `--workers 0` to
disable multiprocessing. Restoring optimizer, scheduler, AMP and RNG states does
not guarantee bitwise-identical persistent-worker augmentation streams.

## 8. Evaluate a trained checkpoint

Find the selected checkpoint in the fold's `completed.json`. Replace the example
checkpoint argument with that actual path:

```bash
python evaluate.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --fold 1 --checkpoint "PATH_TO_BEST.ckpt" --output ../results/wpc_fold1.csv
```

For SJTU, change the dataset and data-root arguments accordingly. Evaluation
writes a per-sample CSV and a `.metrics.json` file.

SROCC/KROCC use raw predictions and MOS. PLCC/RMSE use the loaded base metric
implementation; the reference protocol uses unconstrained five-parameter fitting.
The unfitted correlation is additionally saved as `plcc_raw`. Training selects
the highest held-out-fold SROCC across epochs; no additional inner validation
set is introduced.

Final quality checkpoints are not currently available for download. Train first
using the commands above, then evaluate the resulting best checkpoint. Loading
only Swin pretrained weights does not yield a trained quality model.
