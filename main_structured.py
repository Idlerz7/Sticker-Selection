import os

# Work around protobuf C-extension segfault in some environments.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from utils import try_create_dir
from structured_retrieval import parse_structured_args, run_structured_main


def main() -> None:
    try_create_dir("./logs")
    args = parse_structured_args()
    run_structured_main(args)


if __name__ == "__main__":
    main()
