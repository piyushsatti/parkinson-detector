import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torchaudio
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


PARKINSON_FOLDERS = {
    "28 people with parkinson's disease",
}

CONTROL_FOLDERS = {
    "15 young healthy control",
    "22 elderly healthy control",
}


def _normalize_part(part: str) -> str:
    return part.lower().replace("’", "'")


@dataclass
class Record:
    """Container for a single audio example."""

    utt_id: str
    wav: str
    speaker: str
    label: str
    duration: float


def infer_label(path: Path) -> str:
    """Infer a binary label from the file path."""
    parts = {_normalize_part(p) for p in path.parts}
    if parts & PARKINSON_FOLDERS:
        return "parkinson"
    if parts & CONTROL_FOLDERS:
        return "not_parkinson"
    logger.warning("Falling back to unknown label for path=%s", path)
    return "unknown"


def infer_speaker_id(path: Path) -> str:
    """Derive speaker id from the folder immediately above the wav."""
    return path.parent.name.replace(" ", "_")


def compute_duration(path: Path) -> float:
    """Compute duration in seconds using torchaudio.info."""
    try:
        info = torchaudio.info(str(path))
        if info.sample_rate == 0:
            raise ValueError(f"Invalid sample rate for {path}")
        return info.num_frames / info.sample_rate
    except RuntimeError as exc:
        try:
            import soundfile as sf
        except ImportError as import_exc:
            raise RuntimeError(
                "torchaudio backend not available; install soundfile "
                "(and libsndfile) to read wav metadata."
            ) from import_exc
        with sf.SoundFile(str(path)) as f:
            if f.samplerate == 0:
                raise ValueError(f"Invalid sample rate for {path}")
            return len(f) / f.samplerate


def build_manifest(records: Iterable[Record]) -> Dict[str, Dict]:
    """Convert Record objects into the SpeechBrain JSON manifest structure."""
    manifest = {}
    for rec in records:
        manifest[rec.utt_id] = {
            "wav": rec.wav,
            "length": rec.duration,
            "label": rec.label,
            "speaker": rec.speaker,
        }
    return manifest


def _make_id(wav_path: Path, root: Path) -> str:
    rel = wav_path.relative_to(root)
    return rel.with_suffix("").as_posix().replace("/", "_").replace(" ", "_")


def scan_dataset(root: Path, placeholder: str = "{data_root}") -> List[Record]:
    """
    Walk a dataset folder and collect metadata for all wav files.

    Args:
        root: Root folder containing the audio data.
        placeholder: Replacement token stored in manifests for portability.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    wav_paths = sorted(root.rglob("*.wav"))
    records: List[Record] = []
    for wav_path in wav_paths:
        duration = compute_duration(wav_path)
        label = infer_label(wav_path)
        speaker = infer_speaker_id(wav_path)
        utt_id = _make_id(wav_path, root)
        portable_path = f"{placeholder}/{wav_path.relative_to(root).as_posix()}"
        records.append(
            Record(
                utt_id=utt_id,
                wav=portable_path,
                speaker=speaker,
                label=label,
                duration=duration,
            )
        )
    return records


def _group_by_speaker(records: List[Record]) -> Dict[str, List[Record]]:
    grouped: Dict[str, List[Record]] = {}
    for rec in records:
        grouped.setdefault(rec.speaker, []).append(rec)
    return grouped


def _speaker_label(speaker_records: List[Record]) -> str:
    labels = {r.label for r in speaker_records}
    if len(labels) != 1:
        raise ValueError(f"Mixed labels for speaker: {labels}")
    return labels.pop()


def split_speaker_level(
    records: List[Record], val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[Record]]:
    grouped = _group_by_speaker(records)
    speakers = sorted(grouped.keys())
    speaker_labels = [_speaker_label(grouped[s]) for s in speakers]

    train_spk, test_spk = train_test_split(
        speakers,
        test_size=test_ratio,
        stratify=speaker_labels,
        random_state=seed,
    )
    train_labels = [_speaker_label(grouped[s]) for s in train_spk]
    train_spk, val_spk = train_test_split(
        train_spk,
        test_size=val_ratio / (1 - test_ratio),
        stratify=train_labels,
        random_state=seed,
    )

    splits = {"train": train_spk, "valid": val_spk, "test": test_spk}
    out: Dict[str, List[Record]] = {k: [] for k in splits}
    for split_name, spk_list in splits.items():
        for spk in spk_list:
            out[split_name].extend(grouped[spk])
    return out


def split_file_level(
    records: List[Record], val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[Record]]:
    ids = list(range(len(records)))
    labels = [r.label for r in records]

    train_ids, test_ids = train_test_split(
        ids, test_size=test_ratio, stratify=labels, random_state=seed
    )
    train_labels = [labels[i] for i in train_ids]
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_ratio / (1 - test_ratio),
        stratify=train_labels,
        random_state=seed,
    )

    split_ids = {"train": train_ids, "valid": val_ids, "test": test_ids}
    out: Dict[str, List[Record]] = {k: [] for k in split_ids}
    for split_name, idxs in split_ids.items():
        out[split_name] = [records[i] for i in idxs]
    return out


def summarize_split(records: Dict[str, List[Record]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split, recs in records.items():
        labels = {}
        speakers = set()
        for r in recs:
            labels[r.label] = labels.get(r.label, 0) + 1
            speakers.add(r.speaker)
        summary[split] = {
            "examples": len(recs),
            "speakers": len(speakers),
            **{f"label_{k}": v for k, v in labels.items()},
        }
    return summary


def save_manifest(manifest: Dict[str, Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
