from __future__ import annotations

import os
import sys

from scripts.run_eval import main


if __name__ == "__main__":
    dataset = os.environ.get("GRACE_DATASET", "devign")
    config = os.environ.get("GRACE_CONFIG", "configs/inference.local.yaml")
    input_file = os.environ.get("GRACE_INPUT_FILE", "")

    argv = ["basep.py", "--dataset", dataset, "--mode", "basep", "--config", config]
    if input_file:
        argv.extend(["--input-file", input_file])
    argv.extend(sys.argv[1:])
    sys.argv = argv
    main()

