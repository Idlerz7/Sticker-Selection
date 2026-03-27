"""
Config-driven legacy MM-BERT entrypoint.

This keeps the original `main.py` training pipeline intact, but allows YAML
config files with `extends:` and repeated `--config` overrides.

Run (example):
  python main_mmbbert_yaml.py --config configs/mmbbert/stickerchat_v1.yaml
"""

import os
import sys
from typing import Any, Dict, List, Optional

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from transformers import HfArgumentParser

from main import Arguments, main as run_main
from structured_retrieval_tokens import (
    _config_mapping_to_argv,
    _deep_merge_dict,
    _extract_yaml_config_paths,
    load_structured_token_yaml_with_extends,
)
from utils import try_create_dir


def parse_mmbbert_args(argv: Optional[List[str]] = None) -> Arguments:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_paths, rest = _extract_yaml_config_paths(argv)
    merged: Dict[str, Any] = {}
    for path in config_paths:
        merged = _deep_merge_dict(merged, load_structured_token_yaml_with_extends(path))
    prefix = _config_mapping_to_argv(merged)
    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses(
        args=prefix + rest,
        look_for_args_file=False,
    )
    return args


def main() -> None:
    try_create_dir("./logs")
    args = parse_mmbbert_args()
    run_main(args)


if __name__ == "__main__":
    main()
