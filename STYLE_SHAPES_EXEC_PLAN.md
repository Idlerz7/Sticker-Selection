# Style Shapes Living Execution Plan

Last update: 2026-07-23 16:10 UTC

## Immutable inputs

- Repository HEAD at start: `e9d7f3211972308fcea2481c589ad340c94b1e53`.
- Conda environment: `stickr-select`.
- DSTC VPD: `artifacts/lvpcm/descriptors/dstc/multi_clean.pt`, `[307,256]`.
- StickerChat VPD: `artifacts/lvpcm/descriptors/stickerchat/multi_clean.pt`,
  `[174695,256]`.
- Final CLIP sources are the matching LVPCM `final_clip_clean.pt` bundles.
- DSTC reference bank: `factorized_style_bank.json`.
- StickerChat reference bank:
  `stickerchat/processed_style_kmeans_k384/factorized_style_bank.json`.
- Fixed evaluation candidates are the original `data/` and `stickerchat/processed/` files,
  never group-rebuilt evaluation files.

## Stage ledger

| Stage | State | Gate / next action |
|---|---|---|
| Repository and asset audit | COMPLETE | Hashes and all five fixed candidate files validated |
| Group Bank v1 and builders | COMPLETE | Seven compact banks built and content-hashed |
| Legacy equivalence | PASS | Partition, sampling, prototype score and loss are elementwise equal |
| Weights-only init / order / trace | EXECUTION_BLOCKED | Order/trace code and manifests complete; snapshot GPU command approval failed |
| DSTC formal pilot | RESOURCE_BLOCKED | Requires four equivalent free A800 GPUs |
| StickerChat formal pilot | RESOURCE_BLOCKED | Requires eight equivalent free A800 GPUs |
| Bootstrap and final verdict | NOT_EVALUATED | No formal query scores; no research claim |

At final resource audit fewer than four equivalent A800s were free. Occupied GPUs were not preempted.
The formal matrix is RESOURCE_BLOCKED and no historical or smoke metric is substituted.

## Exact pilot matrix

- DSTC: `llm_original`, `final_clip`, `vpd_multi`, `random_matched`,
  `base_only_reference_sampler`; world size 4.
- StickerChat: `final_clip_pack_original`, `vpd_pack`, `random_pack_matched`,
  `base_only_reference_sampler`; world size 8.

Every dataset-level variant shares the initialization hash, epoch-permutation hash, batch size,
candidate order, and hardware conditions. Each rank writes an atomic negative trace containing
`source_row`, positive, fallback, cross, same, epoch, rank, and membership hash. Merged traces
must cover every expected epoch/source row exactly once.

## Preregistered statistical decision

- VPD over Random: R@1 and MRR confidence-interval lower bounds greater than zero.
- VPD over Final CLIP: positive R@1 and MRR point differences and MRR lower bound above zero.
- DSTC versus LLM: R@1 and MRR lower bounds at least `-0.01`.
- StickerChat versus current grouping: nonnegative points and lower bounds at least `-0.005`.
- VPD versus the reference-sampler base-only control: MRR lower bound above zero and positive
  R@1 point difference.
- No empty group; maximum occupancy at most `max(20%, 2*reference)`; effective group count at
  least half reference; same-group-negative coverage at least 75% and no more than 5 percentage
  points below reference.

No K scan, loss change, candidate change, early stopping, or per-variant tuning is permitted.

