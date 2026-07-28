# VIGEM Instance-Residual

This experiment is isolated from the existing SEMSP and Style Shapes runners.
Activate the environment once, then invoke Python directly:

```bash
conda activate stickr-select

python scripts/vigem/build_instance_residual_assets.py

CUDA_VISIBLE_DEVICES=0 python scripts/vigem/create_init_snapshot.py \
  --config configs/vigem/dstc_vpd_multi_instance_residual.yaml

CUDA_VISIBLE_DEVICES=0 python scripts/vigem/create_init_snapshot.py \
  --config configs/vigem/stickerchat_vpd_pack_fixed_r10_instance_residual.yaml
```

Real-asset smoke tests:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vigem/run_smoke.py \
  --config configs/vigem/dstc_vpd_multi_instance_residual.yaml

CUDA_VISIBLE_DEVICES=0 python scripts/vigem/run_smoke.py \
  --config configs/vigem/stickerchat_vpd_pack_fixed_r10_instance_residual.yaml
```

Formal training uses every GPU listed in `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/vigem/run_instance_residual.py \
  --config configs/vigem/dstc_vpd_multi_instance_residual.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/vigem/run_instance_residual.py \
  --config configs/vigem/stickerchat_vpd_pack_fixed_r10_instance_residual.yaml
```

Final evaluation is single-GPU and writes both full-model and
`without_instance` metrics:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vigem/run_instance_residual.py \
  --config configs/vigem/stickerchat_vpd_pack_fixed_r10_instance_residual.yaml \
  --run-mode test \
  --checkpoint-path artifacts/vigem/pilot/stickerchat/vpd_pack_fixed_r10_instance_residual/final.ckpt
```

Use `--test-data-path` and `--run-output-dir` for the random same-pack R10 and
global-random R20 checks without changing the registered config.

## Pack-relative setwise repair (StickerChat fixed R10 only)

This is an independent experiment. It does not replace the existing
instance-residual runner and it intentionally refuses every R20 path.

```bash
python scripts/vigem/build_pack_relative_assets.py \
  --config configs/vigem/stickerchat_vpd_pack_relative_setwise_r10.yaml

python scripts/vigem/create_pack_relative_init_snapshot.py \
  --config configs/vigem/stickerchat_vpd_pack_relative_setwise_r10.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/vigem/run_pack_relative.py \
  --config configs/vigem/stickerchat_vpd_pack_relative_setwise_r10.yaml
```

Final fixed same-pack Test R10 evaluation is single-GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vigem/run_pack_relative.py \
  --config configs/vigem/stickerchat_vpd_pack_relative_setwise_r10.yaml \
  --run-mode test \
  --checkpoint-path artifacts/vigem/pilot/stickerchat/vpd_pack_relative_setwise_r10/final.ckpt
```

Pass `--baseline-scores PATH` during final evaluation to add the preregistered
10,000-query-bootstrap comparison against the aligned completed VIGEM run.
