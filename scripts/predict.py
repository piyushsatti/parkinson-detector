#!/usr/bin/env python3
"""
Lightweight inference helper for trained checkpoints.
Usage:
  python scripts/predict.py --hparams recipes/parkinsons_binary/xvector/hparams/train.yaml \
      --checkpoint_dir results/xvector/1986/save --data_folder data/raw/italian_parkinson --wav path/to/file.wav
"""
import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from parkinsons_speech.utils import random_crop  # noqa: E402


DEFAULT_LABELS = ["not_parkinson", "parkinson"]


def load_labels(save_folder: Path):
    enc_path = save_folder / "label_encoder.txt"
    if not enc_path.exists():
        return DEFAULT_LABELS
    labels = []
    with open(enc_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                labels.append(parts[0])
    return labels or DEFAULT_LABELS


def build_model(hparams_path: Path, checkpoint_dir: Path, data_folder: Path):
    with open(hparams_path) as fin:
        hparams = load_hyperpyyaml(fin)

    hparams["data_folder"] = str(data_folder)
    hparams["checkpointer"].checkpoints_dir = str(checkpoint_dir)

    modules = hparams["modules"]
    hparams["checkpointer"].recover_if_possible()
    return hparams, modules


def prepare_audio(path: Path, hparams):
    sig, sr = torchaudio.load(path)
    if sr != hparams["sample_rate"]:
        sig = torchaudio.functional.resample(
            sig, orig_freq=sr, new_freq=hparams["sample_rate"]
        )
    sig = random_crop(sig, hparams["sample_rate"], hparams["chunk_duration"])
    sig = sig / torch.clamp(sig.abs().max(), min=1e-6)
    return sig


def forward(modules, hparams, wav: torch.Tensor):
    lens = torch.tensor([1.0])
    if "feature_extractor" in modules:
        feats = modules["feature_extractor"](wav)
        emb = modules["xvector"](feats, lens)
        logits = modules["classifier"](emb)
    else:
        outputs = modules["ssl_model"](wav, lens)
        pooled = hparams["avg_pool"](outputs, lens)
        logits = modules["output_mlp"](pooled.view(pooled.shape[0], -1))
    probs = hparams["log_softmax"](logits).exp().squeeze(0)
    return probs


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single wav file.")
    parser.add_argument("--hparams", required=True, help="Path to HyperPyYAML file used for training.")
    parser.add_argument("--checkpoint_dir", required=True, help="Folder containing saved checkpoints.")
    parser.add_argument("--data_folder", required=True, help="Root of raw data (for manifest placeholders).")
    parser.add_argument("--wav", required=True, help="Path to wav file to classify.")
    args = parser.parse_args()

    hparams_path = Path(args.hparams)
    checkpoint_dir = Path(args.checkpoint_dir)
    data_folder = Path(args.data_folder)
    wav_path = Path(args.wav)

    hparams, modules = build_model(hparams_path, checkpoint_dir, data_folder)
    labels = load_labels(checkpoint_dir)

    wav = prepare_audio(wav_path, hparams).unsqueeze(0)
    probs = forward(modules, hparams, wav)

    top_idx = int(torch.argmax(probs).item())
    print("Prediction:", labels[top_idx])
    for idx, label in enumerate(labels):
        print(f"{label}: {probs[idx].item():.4f}")


if __name__ == "__main__":
    main()
