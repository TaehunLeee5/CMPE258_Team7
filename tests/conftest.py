import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wm.config import load_config  # noqa: E402


@pytest.fixture
def small_cfg():
    """Baseline config shrunk to a CPU-friendly synthetic setup."""
    return load_config(REPO_ROOT / "configs" / "baseline.yaml", [
        "data.synthetic=true",
        "data.num_train_images=4",
        "data.num_val_images=2",
        "data.image_size=32",
        "data.train_samples_per_epoch=16",
        "train.batch_size=4",
        "model.encoder_channels=8",
        "model.decoder_channels=8",
        "experiment.device=cpu",
    ])
