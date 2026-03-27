"""
Structured Factorized entrypoint.

This line uses a two-stage style/expression retriever:
1) style/prototype-driven candidate union
2) learned fusion reranker over MM-BERT + style + expression + graph scores

Run (example):
  python main_structured_factorized.py --config configs/structured_factorized/v4_01_style_expr_two_stage.yaml
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from structured_retrieval_factorized import (
    parse_structured_factorized_args,
    run_structured_factorized_main,
)
from utils import try_create_dir


def main() -> None:
    try_create_dir("./logs")
    args = parse_structured_factorized_args()
    run_structured_factorized_main(args)


if __name__ == "__main__":
    main()
