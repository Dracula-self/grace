from __future__ import annotations

import os
import sys

from scripts.generate_examples import main


if __name__ == "__main__":
    dataset = os.environ.get("GRACE_DATASET", "devign")
    split = os.environ.get("GRACE_SPLIT", "test")
    argv = ["genexample.py", "--dataset", dataset, "--split", split]
    argv.extend(sys.argv[1:])
    sys.argv = argv
    main()

