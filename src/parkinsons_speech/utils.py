import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: os.PathLike) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def random_crop(sig: torch.Tensor, sr: int, max_dur: float) -> torch.Tensor:
    """Crop or pad a waveform to a fixed duration."""
    max_len = int(sr * max_dur)
    if sig.shape[-1] > max_len:
        start = torch.randint(0, sig.shape[-1] - max_len + 1, (1,)).item()
        sig = sig[..., start : start + max_len]
    else:
        pad = max_len - sig.shape[-1]
        sig = torch.nn.functional.pad(sig, (0, pad))
    return sig


def resolve_path(path: str | os.PathLike) -> Path:
    return Path(path).expanduser().resolve()

