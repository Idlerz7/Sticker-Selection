# Style Shapes: VPD Group-Source Fair Pilot Specification

Status: engineering in progress; formal pilot is gated by equivalence checks and GPU availability.

## Scientific contract

The formal model is the `minimal` branch in `structured_retrieval_factorized.py`: MM-BERT,
the shared sticker MLP, instance/expression and group/style heads, learned-feature prototype
pooling, additive score fusion, and the existing losses, optimizer, and scheduler. VPD is an
offline source of group membership only. It is never used as a prototype or a forward input.

The fixed group counts are DSTC `K=85` and StickerChat pack-level `K=384`. The compared group
sources are:

- DSTC: `llm_original`, `final_clip`, `vpd_multi`, `random_matched`.
- StickerChat: `final_clip_pack_original`, `vpd_pack`, `random_pack_matched`.
- Each dataset also has a `base_only_reference_sampler`; this retains reference-bank negative
  sampling and therefore measures the complete group-induced training structure, not a pure
  no-group intervention.

Two isolated optional StickerChat dual-local variants replace prototype-aware cross/same
sampling with one raw same-pack negative plus one distinct random group Final-CLIP top-32
neighbor. `vpd_pack_dual_local_negatives` uses the VPD membership;
`semsp_dual_local_negatives` uses the original SEMSP
`final_clip_pack_original` membership. Both filter singleton raw-pack positives and have no
fallback.

Two further isolated fixed-listwise variants are
`stickerchat_vpd_pack_fixed_same_pack_r10` and
`stickerchat_semsp_fixed_same_pack_r10`. They retain all 320,168 training rows and replace the
legacy three-candidate match objective with one gold plus nine frozen original-pack candidates.
The gold is candidate 0. After removing the gold by normalized external ID, negatives are the
first nine remaining `emoji_mapping.txt` entries in original order. Missing slots use sentinel
ID `-1` backed by one frozen RGB-127 CLIP embedding. Sentinel candidates receive ordinary
MM-BERT and expression scores but a group score of exactly zero.

The raw archive contains two empty `emoji_mapping.txt` files. This upstream defect is handled
declaratively: only for those empty files, `.npy` entries are read in original ZIP member order.
The fallback packs and content hash are recorded in the candidate manifest. Any other missing
pack, sticker mapping, processed-row alignment, or input-hash mismatch is fatal.

Fixed-listwise training performs one query-major vectorized MM-BERT call for `[B,10]` candidates
by flattening to `[B*10]`. Its main loss is 10-way cross entropy. The expression auxiliary uses
only the highest expression-scoring negative, while the main cross entropy continues to give all
nine negatives probability-weighted gradients. Prototype supervision, orthogonality, additive
fusion, parameterization, initialization, optimizer, scheduler, batch size, and epoch order are
unchanged. Candidate chunk 10 is the default; 5, 2, then 1 are permitted only after a recorded
OOM.

All variants of a dataset must strict-load the same weights-only initialization snapshot, use
the same epoch permutations, fixed candidates, optimizer recipe, and hardware allocation.

## Group Bank v1

`style_shapes.group_bank.v1` stores one compact bidirectional partition:

- `dataset`, `group_source`, `num_groups`, `schema_version`;
- `sticker_ids`, aligned `sticker_to_group`, `group_to_members`, and `group_sizes`;
- per-sticker `same_group_neighbors`;
- declarative `cross_group_pool: {type: catalog_minus_group}`;
- feature, clustering, catalog, seed, hashes, configuration, and UTC provenance.

The membership hash is SHA-256 over canonical JSON for the sticker-ID-sorted sequence of
`[sticker_id, group_id]`. Group IDs must be dense in `[0,K)`, all groups non-empty, and each
sticker must occur exactly once. Writes are atomic. Completed outputs with incompatible input
hashes are never silently overwritten.

`FactorizedStyleBank.from_json` accepts both legacy banks and Group Bank v1. Compact records do
not repeat full group membership; prototype records remain the single source of group members.

## Construction

DSTC spherical Lloyd K-means uses all 307 normalized vectors, shared initial sticker-ID sets,
seed 2021, 10 starts, 100 iterations, and tolerance `1e-6`. `random_matched` exactly preserves
the VPD group-size multiset.

StickerChat aggregates the 174,695 sticker descriptors into 3,516 indivisible pack centroids.
The reference source reproduces the old normalized-centroid algorithm for 40 iterations with
seed 20260330. VPD changes only the descriptor source and reuses the same initial pack IDs.
Random pack matching uses seed 2021 and greedily fills VPD sticker-count targets while preserving
packs and 384 non-empty groups.

New-source same-group neighbors are chosen by final-CLIP cosine similarity inside the new group:
top 5 for DSTC and top 32 for StickerChat. Original sources import the original neighbor lists.

## Gates and reporting

No formal training may begin until legacy/new membership, sampling, scores, and losses are
equivalent for DSTC, and the old StickerChat pack-to-K384 membership and legacy/new forward/loss
are exact. DSTC keeps the legacy `data/validation_pair_with_cand.json`. StickerChat's primary R10
uses one gold plus nine unique negatives from the original filename-derived pack; only a pack
shortfall is filled from the global catalog, preserving all 10,000 queries. StickerChat R20 stays
fully global-random. The previous global R10 remains frozen as a reference protocol. Every group
source uses the same candidate files and ordering.

Training uses seed 2021, 10 epochs, per-device batch 16, lambdas 0.3/0.4/0.5, AdamW betas
0.9/0.98, weight decay 0.2, existing two learning rates and cosine schedule, no warmup or early
stopping. DSTC refresh is exact (`0`), StickerChat refresh is `500`. Final checkpoints alone are
used for formal evaluation.

The fixed-listwise primary evaluation is the clean fixed same-pack Test R10. The existing random
same-pack Test R10 and global-random Test R20 are mandatory generalization checks. Every
single-positive score export reports both MRR and MAP and asserts `MAP == MRR`. Old checkpoints
must also be evaluated on the clean fixed R10 without retraining so candidate-protocol effects
are separated from ten-candidate training effects.

Paired query bootstrap uses 10,000 resamples, seed 2021. The preregistered efficacy, size,
coverage, and non-inferiority gates are exactly those in `STYLE_SHAPES_EXEC_PLAN.md`. A failed
engineering gate or scientific gate gives `STOP`. Insufficient free resources gives
`RESOURCE_BLOCKED`; smoke or historical metrics never substitute for formal results.
