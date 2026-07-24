# Style Shapes execution commands

Run every Python command in Conda environment `stickr-select` from the repository root.
All commands append start/finish provenance to
`artifacts/style_shapes/logs/command_history.jsonl`.

```bash
conda run -n stickr-select python \
  scripts/style_shapes/build_stickerchat_same_pack_candidates.py
conda run -n stickr-select python \
  scripts/style_shapes/build_stickerchat_dual_negative_assets.py
conda run -n stickr-select python \
  scripts/style_shapes/build_stickerchat_dual_negative_assets.py \
  --group-bank artifacts/style_shapes/groups/stickerchat/final_clip_pack_original/group_bank.json \
  --eligibility-output artifacts/style_shapes/negative_sampling/stickerchat_same_pack_plus_semsp_top32/eligible_rows.json \
  --permutation-output artifacts/style_shapes/permutations/stickerchat_semsp_dual_local_seed2021_ws8.json
conda run -n stickr-select python scripts/style_shapes/audit_repo.py
conda run -n stickr-select python scripts/style_shapes/build_groups.py \
  --config configs/style_shapes/dstc_groups.yaml
conda run -n stickr-select python scripts/style_shapes/build_groups.py \
  --config configs/style_shapes/stickerchat_groups.yaml
conda run -n stickr-select python scripts/style_shapes/check_bank_equivalence.py
```

Initialization snapshots are weights-only and must be created once, after the equivalence gate:

```bash
CUDA_VISIBLE_DEVICES=<one-free-gpu> conda run -n stickr-select python \
  scripts/style_shapes/create_init_snapshot.py \
  --base-config configs/structured_factorized/v6_00_minimal_core.yaml \
  --group-bank artifacts/style_shapes/groups/dstc/llm_original/group_bank.json \
  --output artifacts/style_shapes/init/dstc_seed2021.ckpt

CUDA_VISIBLE_DEVICES=<one-free-gpu> conda run -n stickr-select python \
  scripts/style_shapes/create_init_snapshot.py \
  --base-config configs/structured_factorized/stickerchat_v6_minimal_core.yaml \
  --group-bank artifacts/style_shapes/groups/stickerchat/final_clip_pack_original/group_bank.json \
  --output artifacts/style_shapes/init/stickerchat_seed2021.ckpt
```

Run one formal training source only when the required equivalent devices are all free:

```bash
CUDA_VISIBLE_DEVICES=<four-free-gpus> conda run -n stickr-select python \
  scripts/style_shapes/run_pilot.py \
  --config configs/style_shapes/dstc_vpd_multi.yaml

CUDA_VISIBLE_DEVICES=<eight-free-gpus> conda run -n stickr-select python \
  scripts/style_shapes/run_pilot.py \
  --config configs/style_shapes/stickerchat_vpd_pack.yaml
```

The other seven source configs are in `configs/style_shapes/`. Do not run variants in
parallel or on unequal hardware.

The optional StickerChat dual-local negative variants are isolated from the registered
group-source pilot. Both filter the 292 training rows whose original pack is a singleton,
then train with one raw same-pack negative and one distinct random member of the positive
sticker's group Final-CLIP top-32 list:

```bash
CUDA_VISIBLE_DEVICES=<visible-gpus> conda run -n stickr-select python \
  scripts/style_shapes/run_pilot.py \
  --config configs/style_shapes/stickerchat_vpd_pack_dual_local_negatives.yaml

CUDA_VISIBLE_DEVICES=<visible-gpus> conda run -n stickr-select python \
  scripts/style_shapes/run_pilot.py \
  --config configs/style_shapes/stickerchat_semsp_dual_local_negatives.yaml
```

The same-pack negative occupies the factorized `same` slot and receives `expr_rank_loss`;
the VPD or SEMSP top-32 negative occupies the `cross` slot. The original dataset
`neg_img_id` is recorded as provenance but never used as a fallback.

After a source finishes, run fixed-candidate evaluation on one GPU. DSTC keeps the legacy SEMSP
R10 file. StickerChat uses same-pack R10 (global fallback only for the 163 validation / 164 test
queries whose original pack is too small) and fully global-random R20. The previous global R10
files remain unchanged as reference candidates.

```bash
CUDA_VISIBLE_DEVICES=<one-free-gpu> conda run -n stickr-select python \
  scripts/style_shapes/run_pilot.py \
  --config configs/style_shapes/stickerchat_vpd_pack.yaml \
  --run-mode test \
  --checkpoint-path artifacts/style_shapes/pilot/stickerchat/vpd_pack/final.ckpt \
  --test-data-path stickerchat/processed/release_test_u_sticker_format_int_with_cand_same_pack_r10.json

CUDA_VISIBLE_DEVICES=<one-free-gpu> conda run -n stickr-select python \
  scripts/style_shapes/run_pilot.py \
  --config configs/style_shapes/stickerchat_vpd_pack.yaml \
  --run-mode test \
  --checkpoint-path artifacts/style_shapes/pilot/stickerchat/vpd_pack/final.ckpt \
  --test-data-path stickerchat/processed/release_test_u_sticker_format_int_with_cand_r20.json
```

Validate negative traces before admitting a run, then compute paired bootstrap comparisons
from the exported query-score files. A missing trace, missing protocol, or non-final checkpoint
is a STOP, not a partial result.

For the dual-local variant, validate trace coverage against the frozen eligible source rows:

```bash
conda run -n stickr-select python scripts/style_shapes/validate_traces.py \
  --trace-glob 'artifacts/style_shapes/pilot/stickerchat/vpd_pack_dual_local_negatives/negative_trace/rank_*.jsonl' \
  --epochs 10 \
  --membership-hash b703e4be8c9899a54a621f574bfa96cac18f773a4fbad2632d4cedf5e3de758f \
  --eligibility-manifest artifacts/style_shapes/negative_sampling/stickerchat_same_pack_plus_vpd_top32/eligible_rows.json \
  --pack-metadata stickerchat/processed/sticker_metadata.json \
  --group-bank artifacts/style_shapes/groups/stickerchat/vpd_pack/group_bank.json \
  --output artifacts/style_shapes/pilot/stickerchat/vpd_pack_dual_local_negatives/negative_trace/validation.json
```
