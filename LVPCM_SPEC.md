# LVPCM Method Specification

This document freezes the implementation and evaluation contract for the LVPCM study.  All
new code and generated material lives under `lvpcm/`, `scripts/lvpcm/`,
`configs/lvpcm/`, `tests/lvpcm/`, and `artifacts/lvpcm/`.

## Frozen assets and representations

- Runtime: Conda environment `stickr-select`, seed 2021, no downloaded models or new
  dependencies.
- DSTC neutral model:
  `logs/clip/lightning_logs/version_6/checkpoints/epoch=7-step=211575.ckpt`.
- StickerChat neutral model: unavailable.  A4 and Stage B are BLOCKED for StickerChat until a
  strict-loadable Chinese MM-BERT checkpoint with 32 speaker tokens is supplied.
- CLIP: `ckpt/clip-ViT-B-32/0_CLIPModel`.  Images use the first frame, white alpha
  compositing, RGB conversion, and the local 224 x 224 CLIP processor.
- Visual patch distribution (VPD): 1-based blocks 3/6/9 are
  `hidden_states[3]`, `hidden_states[6]`, and `hidden_states[9]`.  Token 0 is removed;
  each layer contains 49 patch vectors of width 768.  Per-layer patch mean and population
  standard deviation are concatenated.  Single-layer VPD uses block 6; multi-layer VPD uses
  blocks 3/6/9.
- Each VPD family fits its own `StandardScaler` and randomized 256-dimensional PCA on the
  frozen training catalog only (`whiten=False`, `random_state=2021`) and then L2 normalizes.
  The final CLIP baseline L2 normalizes the existing 512-dimensional caches.
- DSTC pair cache `h` is BERT `pooler_output` (`[Q,N,768]`) and `b` is positive-class logit
  (`[Q,N]`).

## Frozen catalogs

- DSTC train catalog: union of positive and negative IDs in `data/train_pair.json` (285 IDs).
- StickerChat train catalog: union of positive and original same-pack negative image IDs in
  `stickerchat/release_train_u_sticker_format.json`, mapped by
  `stickerchat/processed/img2id.json` (54,268 IDs).  Later synthetic global negatives are
  excluded from scaler, PCA, and the training graph.

## Local presentation context (LPC)

Cosine top-10 nearest neighbors are computed exactly with deterministic `(similarity, integer
ID)` tie resolution.  Training nodes exclude themselves and retain only mutual edges.  A
non-training query searches only the training index and is reciprocal with training node `j`
iff the query would enter `j`'s top-10 pool when added alone.  Validation/test queries never
communicate with one another.  LPC is the normalized mean of retained VPD neighbors; an empty
neighborhood explicitly falls back to the query VPD.

## Perturbations and audits

Perturbations are 75% bicubic downsample/restore, JPEG quality 75, and alpha bounding-box crop.
The crop is applied and scored only where a transparent margin exists and the crop is safe:
the non-zero-alpha bounding box must differ from the full canvas, be at least 8 x 8 pixels, and
retain at least 25% of the original canvas area.
StickerChat reports ordinary and canonical-RGB-exact-duplicate-filtered, pack-macro R@1/5/10.
DSTC reports full neighbor diagnostics.  Leakage diagnostics cover exact image duplicates,
pack, and existing OCR exact-text/character overlap.  Independent IP/character metadata is
unavailable.

A4 uses a fixed dialogue-group 80/20 split and L2 logistic regression to compare controls with
controls plus presentation variables.  Both probes use standardized inputs, `C=1`, balanced
class weights, and seed 2021.  Gold-to-predicted descriptor distance is descriptive
only.  Ten thousand query bootstraps estimate uncertainty; incremental support requires held-out
AUROC gain >= 0.01 and a positive 95% lower bound.

## Stage-A gate

1. Multi-layer VPD must beat final CLIP and single-layer VPD on duplicate-filtered pack-macro
   R@10 with a positive paired-bootstrap lower bound, without reversing R@1 or R@5.
2. LPC must improve the mean three-perturbation Jaccard@10 over multi-layer single-image VPD by
   >= 0.02 with positive lower bound.  Each dataset needs reciprocal coverage >= 75% and fallback
   <= 25%.
3. LPC may increase exact-duplicate or exact-OCR top-10 share by at most five percentage points
   and must retain >= 25% cross-pack neighbors.
4. A core VPD, locality, or serious leakage failure yields STOP.  A1-A3 passing while A4 is
   blocked by an audited missing asset yields CONDITIONAL GO.  GO requires supporting A4 on both
   datasets.

## Stage B

Stage B runs only after GO or CONDITIONAL GO, and only for strict-loadable datasets.  It compares
base, unconditional, final-CLIP, single-VPD, local-LPC, and shuffled-LPC variants on shared pair
caches, candidates, initialization policy, and saved epoch permutations.  Conditional scorers
use non-affine LayerNorm, `A:[16,768]` (Xavier), `C:[16,256]` (zero), and
`sigma_b*tanh(<A LN(h), C z>/sqrt(16))`.  The unconditional control uses `U:[21,768]` and
zero `v:[21]`.  Training is ten epochs, batch 256, AdamW lr 1e-3, no weight decay or warmup,
cosine decay, listwise cross entropy, and final-checkpoint evaluation.

The formal Stage-B gate requires Local to beat Unconditional, Final CLIP, and Single VPD in MRR
with positive paired-bootstrap lower bounds; Local must improve R@1 over Unconditional; Shuffled
must remove at least 75% of the Local gain; and both StickerChat global and same-pack protocols
must support the result.  Missing required protocols or failed comparisons yields STOP.

## Reproducibility and output policy

CLIs record normalized arguments, UTC time, Git state, environment, input hashes, and exit code.
Artifacts are atomically written and hashed.  Matching manifests are resumable; incompatible
inputs or configurations are never silently overwritten.  Smoke outputs are labeled and are
excluded from research tables.  The living execution record is `LVPCM_EXEC_PLAN.md`.
