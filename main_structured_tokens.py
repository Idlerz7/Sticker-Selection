"""
Structured Token V2 entrypoint (early injection of structured embeddings into BERT inputs).

Training / losses mirror `structured_retrieval.py`; only the multimodal input layout changes.

Config-driven runs (recommended):
  python main_structured_tokens.py --config configs/structured_tokens/v2_01_expr.yaml

YAML may set `extends: _base.yaml` (relative to the YAML file) for shared defaults.
Pass multiple `--config a.yaml b.yaml` to deep-merge (later overrides earlier).
Any extra CLI flags after the config paths override YAML.
"""

import os

# Work around protobuf C-extension segfault in some environments.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import distutils_tensorboard_shim  # noqa: F401

from utils import try_create_dir
from structured_retrieval_tokens import (
    parse_structured_token_args,
    run_structured_tokens_main,
)


def main() -> None:
    try_create_dir("./logs")
    args = parse_structured_token_args()
    run_structured_tokens_main(args)


if __name__ == "__main__":
    main()
