"""Train the watermarking model.

    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --set train.epochs=1 data.synthetic=true
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import load_config  # noqa: E402
from wm.trainer import Trainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Config overrides, e.g. train.epochs=5 experiment.name=debug")
    args = parser.parse_args()
    Trainer(load_config(args.config, args.set)).fit()


if __name__ == "__main__":
    main()
