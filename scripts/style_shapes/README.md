# Style Shapes execution commands

Run every Python command in Conda environment `stickr-select` from the repository root.
All commands append start/finish provenance to
`artifacts/style_shapes/logs/command_history.jsonl`.

```bash
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

After a source finishes, run fixed-candidate evaluation on one GPU. DSTC has one R10 file.
StickerChat requires both commands:

```bash
CUDA_VISIBLE_DEVICES=<one-free-gpu> conda run -n stickr-select python \
  scripts/style_shapes/run_pilot.py \
  --config configs/style_shapes/stickerchat_vpd_pack.yaml \
  --run-mode test \
  --checkpoint-path artifacts/style_shapes/pilot/stickerchat/vpd_pack/final.ckpt \
  --test-data-path stickerchat/processed/release_test_u_sticker_format_int_with_cand_r10.json

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

