# Protocol reference and release boundary

This package provides additions and reader runners, not a copy of the base model
or metric implementation. Runners import DA-Net externally. See READER_GUIDE_zh.md.

| Setting | WPC | SJTU-PCQA |
|---|---|---|
| Folds | 5 | 9 |
| Samples | 740 | 378 |
| Train/held-out per fold | 592 / 148 | 336 / 42 |
| Seed / epochs / batch | 42 / 100 / 4 | 42 / 100 / 4 |
| Train / held-out patches | 8 / 32 | 8 / 32 |
| MOS divisor | 100 | 10 |
| Local XYZRGB normalization | off | on, as historical base loader |
| Factor mapping | WPC parameter lookup | SJTU rank6 |
| Factor weight / beta | 0.02 each / 0.1 | 0.02 each / 0.1 |

GTFS init uses seed2001; LGQC init uses seed3501 and a zero-initialized final
bias-free linear layer. The global call uses seed3501001+epoch*100000+step in an
isolated RNG context. SJTU preserves CUDA RNG during extension construction.
Optimizer: AdamW lr5e-5, weight_decay1e-5; StepLR step_size5 gamma0.9.

The base constructor/forward, original loss weights/schedule, augmentation,
local input arithmetic and fitted metrics remain external. Do not substitute
another implementation while claiming exact historical reproduction.

Split recipes preserve partition membership and row order, including corrected,
content-disjoint WPC fold5. MOS is supplied externally.
Selection is highest held-out SROCC each epoch; no separate inner validation.
SROCC/KROCC use raw scores; historical PLCC/RMSE used unconstrained five-parameter
fitting. The selected checkpoint, fitting convention and raw predictions matter.

Global cache: six 448x448 orthographic nearest-surface views, explicit occupancy,
7x7 grid, 64x64 cells, 32x32 crops, counts8/8/8/8/8/9, mosaic224x224.
Training mosaics use an independent worker RNG; evaluation is name-deterministic.
The shared encoder remains trainable through the global residual.

Historical WPC used upstream local PNG caches; their original T-Net weights were
unavailable. Historical SJTU T-Net used all unlabelled geometry, including quality
held-out content. The new incremental integration has not undergone full GPU
retraining against a freshly downloaded official revision.
