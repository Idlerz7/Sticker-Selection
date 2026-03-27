"""
Structured Residual V3 entrypoint.

This line conditions the legacy sticker token with structured residuals rather
than appending extra struct tokens to the BERT sequence.

Run (example):
  python main_structured_residual.py --config configs/structured_residual/v3_01_expr_residual.yaml
"""

import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from structured_retrieval_residual import (
    parse_structured_residual_args,
    run_structured_residual_main,
)
from utils import try_create_dir


def _should_dispatch_factorized(argv) -> bool:
    factorized_flags = {
        "--factorized_bank_path",
        "--factorized_candidate_union_topk",
        "--factorized_eval_compare_mmbert",
        "--factorized_fusion_hidden_dim",
        "--factorized_log_interval",
        "--factorized_mmbert_branch_topk",
        "--factorized_proto_expand_per_proto",
        "--factorized_proto_logit_scale",
        "--factorized_pseudo_label_path",
        "--factorized_style_proto_topk",
        "--factorized_style_recall_topk",
        "--factorized_use_mmbert_branch",
        "--lambda_style_proto",
        "--train_cross_proto_negatives",
        "--train_same_proto_negatives",
    }
    return any(
        ("structured_factorized" in str(arg)) or (str(arg) in factorized_flags)
        for arg in argv
    )


def main() -> None:
    try_create_dir("./logs")
    argv = sys.argv[1:]
    if _should_dispatch_factorized(argv):
        from structured_retrieval_factorized import (
            parse_structured_factorized_args,
            run_structured_factorized_main,
        )

        print(
            "[EntryPointRedirect] Detected factorized config/flags in residual entrypoint; "
            "dispatching to structured factorized runner.",
            file=sys.stderr,
        )
        args = parse_structured_factorized_args(argv)
        run_structured_factorized_main(args)
        return

    args = parse_structured_residual_args(argv)
    run_structured_residual_main(args)


if __name__ == "__main__":
    main()
