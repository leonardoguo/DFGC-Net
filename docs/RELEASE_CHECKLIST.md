# Before publishing

- [x] License our contributed code under MIT; preserve third-party resource boundaries (LICENSE/NOTICE.md).
- [ ] Add final author information and paper citation; do not invent a DOI.
- [ ] Confirm no base implementation, data or copied weights entered the repository.
- [ ] Pin/record the supported DA-Net revision and source hashes.
- [x] Perform a real full-backbone GPU train/validation smoke (WPC/SJTU fold1; see VERIFICATION.md).
- [ ] Verify dataset, Swin and historical cache download links remain accessible.
- [ ] Check permissions separately before distributing weights/caches.
- [ ] Keep normalization, selection, fitting and transductive T-Net description accurate.
- [ ] Keep outputs, MOS, credentials, local paths and raw/cache assets out of Git.

The reader guide provides complete commands. Runtime imports external components;
it is not a distribution of those components. No LS-PCQA/ablation package promised.
