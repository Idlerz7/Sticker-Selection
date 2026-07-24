# Style Shapes Living Execution Plan

Last update: 2026-07-24 21:25 UTC

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
- DSTC fixed evaluation candidates remain the original SEMSP
  `data/validation_pair_with_cand.json`. StickerChat primary R10 is pack-derived with global
  fallback only when a pack has fewer than nine alternatives; R20 remains fully global-random,
  and the old global R10 remains a reference. Candidates are frozen once and never depend on
  the compared K384 grouping.

## Stage ledger

| Stage | State | Gate / next action |
|---|---|---|
| Repository and asset audit | COMPLETE | Legacy DSTC, StickerChat same-pack R10/global R20, and reference global R10 hashes validated |
| Group Bank v1 and builders | COMPLETE | Seven compact banks built and content-hashed |
| Legacy equivalence | PASS | Partition, sampling, prototype score and loss are elementwise equal |
| Weights-only init / order / trace | EXECUTION_BLOCKED | Order/trace code and manifests complete; snapshot GPU command approval failed |
| DSTC formal pilot | RESOURCE_BLOCKED | Requires four equivalent free A800 GPUs |
| StickerChat formal pilot | RESOURCE_BLOCKED | Requires eight equivalent free A800 GPUs |
| Bootstrap and final verdict | NOT_EVALUATED | No formal query scores; no research claim |
| Fixed same-pack R10 assets | COMPLETE | 320,168/10,000/10,000 rows; gray counts and hashes frozen |
| Fixed-listwise VPD/SEMSP engineering | PASS | Batch-16 160-pair forward/backward and strict reload pass on both banks |
| Fixed-listwise formal training | NOT_STARTED | Run the two independent 10-epoch configs; smoke is not a result |

At final resource audit fewer than four equivalent A800s were free. Occupied GPUs were not preempted.
The formal matrix is RESOURCE_BLOCKED and no historical or smoke metric is substituted.

## Exact pilot matrix

- DSTC: `llm_original`, `final_clip`, `vpd_multi`, `random_matched`,
  `base_only_reference_sampler`; world size 4.
- StickerChat: `final_clip_pack_original`, `vpd_pack`, `random_pack_matched`,
  `base_only_reference_sampler`; world size 8.

The isolated `stickerchat_vpd_pack_dual_local_negatives` and
`stickerchat_semsp_dual_local_negatives` special runs use the same initialization,
optimization recipe, evaluation candidates, 319,876 eligible rows, and 292 singleton-pack
exclusions. They differ only in the group bank used for the second top-32 negative: VPD
membership versus original SEMSP `final_clip_pack_original` membership. Neither is part of
the preregistered group-source comparison matrix.

The isolated fixed-listwise matrix contains:

- `stickerchat_vpd_pack_fixed_same_pack_r10`, using the VPD pack bank;
- `stickerchat_semsp_fixed_same_pack_r10`, using the original SEMSP
  `final_clip_pack_original` bank.

Both reuse the same 320,168-row permutation, seed-2021 initialization, fixed candidate manifest,
batch size 16, and optimizer recipe. Frozen asset counts are Train
`320,168 / 5,803 gray rows / 27,828 gray slots`, Validation
`10,000 / 163 / 826`, and Test `10,000 / 164 / 776`. The candidate builder records two raw
empty-mapping packs and their ZIP-member-order fallback rather than hiding the upstream defect.
Engineering smoke passed for both group banks with 160 pairs in one vectorized forward,
nonzero backward gradients, unchanged state-dict keys/parameter count, and strict checkpoint
reload. No formal metric has been generated.

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

After the preregistered candidate-protocol correction above, no K scan, loss change, further
candidate change, early stopping, or per-variant tuning is permitted.
