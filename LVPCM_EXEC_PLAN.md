# LVPCM Living Execution Plan

Last updated: 2026-07-21 UTC

## Frozen decisions

- Environment: `/data-store/zhangziyun/miniconda3/envs/stickr-select`
- Seed: 2021
- DSTC checkpoint: `logs/clip/lightning_logs/version_6/checkpoints/epoch=7-step=211575.ckpt`
- CLIP: `ckpt/clip-ViT-B-32/0_CLIPModel`
- StickerChat A4/Stage B: BLOCKED (no strict-loadable 32-speaker-token neutral checkpoint)
- Optional covariance sketch and full-catalog transductive LPC: NOT RUN

## Stage state

| Item | State | Evidence / next action |
|---|---|---|
| Repository audit | COMPLETE | DSTC strict-load PASS; StickerChat neutral checkpoint BLOCKED |
| Unit and integration tests | **PASS** | Final unittest discovery: 20/20 passed |
| A1 VPD | PASS | Multi-layer beats final and single on duplicate-filtered pack-macro R@10 |
| A2 LPC | FAIL | DSTC stability delta -0.0474, CI [-0.0698, -0.0254] |
| A3 descriptor/leakage audit | PASS | Duplicate/OCR deltas within 5pp; StickerChat cross-pack 64.55% |
| A4 DSTC | COMPLETE / NO SUPPORT | AUROC gain -0.00117, CI crosses zero |
| A4 StickerChat | BLOCKED | Missing neutral checkpoint; never substitute old English model |
| Stage-A verdict | **STOP** | Locality stability gate failed on DSTC |
| Stage B | **NOT RUN** | Stage-A STOP forbids training |

## Commands and outcomes

Machine-readable command records are appended by every LVPCM CLI to
`artifacts/lvpcm/logs/command_history.jsonl`.  Test, extraction, and training logs are stored under
`artifacts/lvpcm/logs/`.  This table is updated at stage boundaries; it is not a substitute for
the immutable command log.

## Known risks and blockers

- StickerChat has a 174,695-image bank and requires substantial GPU extraction and exact-neighbor
  compute; all long jobs use manifests and resumable shards.
- The working tree was already ahead of its remote by four commits and contained unrelated
  untracked files.  They are preserved unchanged.
- No new OCR, identity metadata, FAISS, pandas, pytest, model, or external service will be added.
- StickerChat same-pack candidate reconstruction is additionally BLOCKED: 16 validation rows and
  5 test rows have `neg_img_id=None`, so the required original same-pack negative cannot be
  mapped or safely invented.

## Recorded implementation findings

- The first candidate-manifest run failed on the missing raw StickerChat negative and recorded
  exit code 1.  The corrected audit behavior preserves DSTC/global manifests and writes explicit
  same-pack BLOCKED evidence.
- The first real smoke exposed the old Pillow API; the compatibility alias retains bicubic and
  LANCZOS algorithms.  The rerun passed and is explicitly marked `research_eligible:false`.
- Cross-query BERT batching changed CUDA floating-point results (`h` max absolute difference
  0.00247).  It was rejected.  Exact-length buckets now execute one native candidate group per
  forward; the persisted DSTC validation cache matches legacy `h` and `b` with max error 0.
- The first sequential StickerChat extractor was interrupted after clean progress and replaced by
  content-addressed contiguous shards.  All accepted shard processes exited 0 and their merged ID
  order exactly reconstructed the 174,695-ID catalog.  Two incomplete temporary memmaps were
  removed after clean shard validation.
- Alpha crop initially performed redundant source decoding.  That process was stopped and its
  failure recorded; the single-decode implementation restarted from ID 0 and its only accepted
  shard exited 0.

## Final Stage-A measurements

- StickerChat multi minus final CLIP duplicate-filtered pack-macro R@10: +0.0537,
  95% CI [0.0493, 0.0581]; R@1/R@5 deltas +0.0781/+0.0666.
- StickerChat multi minus single-layer VPD R@10: +0.0105,
  95% CI [0.0085, 0.0124]; R@1/R@5 deltas +0.0141/+0.0122.
- DSTC LPC coverage/fallback: 94.79%/5.21%; stability delta -0.0474,
  95% CI [-0.0698, -0.0254].
- StickerChat LPC coverage/fallback: 86.29%/13.71%; stability delta +0.0493,
  95% CI [0.0481, 0.0505].
- DSTC A4 control AUROC 0.66844 versus control+presentation 0.66728; gain -0.00117,
  95% CI [-0.01206, 0.00981].
- StickerChat alpha-bbox crop had 0 eligible images under the frozen safety rule; its crop metric
  is explicitly `not_applicable`, while resize and JPEG remain in the stability average.

## Verdict discipline

No exploratory adjustment changes the frozen gate.  A smoke result is never promoted into a
research comparison.  Failures and incomplete protocols remain visible in reports and in the
artifact manifest.
