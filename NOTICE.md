# Scope and attribution

This repository distributes our GTFS/LGQC additions, dataset-specific factor
mappings, occupancy-based global views/mosaics, integration hooks and our own
reader-facing preparation/training/evaluation orchestration.

DA-Net must be obtained separately from https://github.com/Rphone/DA-Net.
No DA-Net base encoder, PointNet++/Swin implementation, T-Net/APPM architecture,
FDM/FMM/VQ/reconstruction implementation, original losses/metrics, local patch
algorithm or projection implementation is bundled. Runners call those components
from the user's external checkout; they do not copy or patch the base source.
The original base forward is executed, not reimplemented here.

Compact multi-view mosaic organization is inspired by GMS-3DQA. Our explicit
occupancy renderer/sampler, supervision additions and orchestration are provided
here as our implementation. Attribution is not a software license.

No external source/data/MOS/weights are downloaded automatically or relicensed.
MOS/split tables and generated assets are created locally from user-obtained
metadata. Published split JSON files contain ordered sample identifiers only.
External assets remain subject to their owners' terms.

The code contributed in this repository is licensed under the MIT License in
LICENSE, copyright (c) 2026 DFGC-Net contributors. Preserve that copyright and
license notice when copying or redistributing this code.

This permission applies to our contributions only. It does not license the
separately obtained DA-Net source, datasets/MOS, caches, pretrained weights or
other external dependencies. Their respective terms remain in force, and this
repository does not grant additional rights to those resources.
