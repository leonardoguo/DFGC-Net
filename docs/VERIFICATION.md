# Tests and validation scope

Run `python -B -m unittest discover -s tests -v`.

Local tests cover factor-head initialization/counts, zero masks, gradient
propagation, residual identity, hook cleanup, strict synthetic reload, occupied
sampling, WPC/SJTU mappings, split-name coverage, preservation of local tensors,
external-MOS joins and SJTU Final_MOS.mat conversion to all nine folds.

Additional integration checks with a separately obtained base implementation
covered parameter initialization, train/eval predictions, losses, gradients,
checkpoint saving and resume. These checks used lightweight backbone substitutes
to isolate the integration and training-control logic.

## Full-backbone GPU smoke tests

WPC fold 1 and SJTU fold 1 passed on an NVIDIA GeForce RTX 4060 Laptop GPU using
the complete Swin-Tiny and PointNet++ encoders, GTFS and LGQC (37,930,621 parameters).
Environment: Python3.9, PyTorch2.2.0+cu118, Lightning2.2.1, PyTorch3D0.7.7.
The tests used actual existing local/global caches, FP16 AMP, batch4 with8 local
patches/cloud during training, and batch1 with32 patches/cloud during validation.
Each test ran four training and four validation batches.

| Dataset | Optimizer updates | Model state finite | Optimizer state finite | Result |
|---|---:|---|---|---|
| WPC fold1 | 3 | yes | yes | passed |
| SJTU fold1 | 4 | yes | yes | passed |

WPC's first AMP update was skipped while the loss scale adjusted. A one-batch
forward/validation run would therefore not have verified actual training updates.
The smoke runner now defaults to four batches and checks optimizer updates plus
finite states, writing smoke_report.json under its separate _smoke run directory.
Lightning may clear gradient tensors during teardown; absent final gradients are
recorded as null, not treated as direct evidence of finite gradients.

Machine-readable summary: [GPU_SMOKE.json](GPU_SMOKE.json).

These are functional smoke tests, not convergence, quality-metric or throughput
benchmarks. No new complete GPU cross-validation or full raw-data regeneration
validation has been performed. Different upstream revisions require interface
checks. Source hashes are recorded to prevent mixing changed recipes in a run.
