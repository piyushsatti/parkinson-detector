#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from parkinsons_speech import data_prep  # noqa: E402
from parkinsons_speech.utils import ensure_dir, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SpeechBrain manifests for Parkinsons dataset.")
    parser.add_argument("--data_root", required=True, help="Root folder containing the wav files.")
    parser.add_argument("--out_dir", default="data/manifests", help="Where to write manifest json files.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument(
        "--split_by",
        choices=["speaker", "file"],
        default="speaker",
        help="Use speaker-level grouping or file-level stratification.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    records = data_prep.scan_dataset(Path(args.data_root))

    if args.split_by == "speaker":
        split = data_prep.split_speaker_level(records, args.val_ratio, args.test_ratio, args.seed)
        # Sanity check: no overlap in speakers
        for a in ("train", "valid", "test"):
            for b in ("train", "valid", "test"):
                if a >= b:
                    continue
                assert set(r.speaker for r in split[a]).isdisjoint(
                    set(r.speaker for r in split[b])
                ), f"Speaker overlap between {a} and {b}"
    else:
        split = data_prep.split_file_level(records, args.val_ratio, args.test_ratio, args.seed)

    for name, subset in split.items():
        manifest = data_prep.build_manifest(subset)
        data_prep.save_manifest(manifest, out_dir / f"{name}.json")

    summary = data_prep.summarize_split(split)
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote manifests to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
