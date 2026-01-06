# Dataset notes

The pipeline expects the Italian Parkinson Voice & Speech dataset arranged locally with speaker folders and WAV files. Example layout after extraction:
```
data/
  raw/
    italian_parkinson/
      28 people with parkinson's disease/
        speaker_01/
          file1.wav
      15 young healthy control/
      22 elderly healthy control/
```

Key details:
- Place the extracted archive under `data/raw/italian_parkinson` (the default `DATA_ROOT` used by Make targets).
- `scripts/prepare_manifests.py` walks all `*.wav` files, infers labels from the parent folders above each speaker, and emits manifests in `data/manifests/`.
- Speaker IDs come from the immediate parent directory of each WAV (spaces are replaced with underscores).

Use `make download` to fetch and extract the archive automatically, or manually download and place files in the same structure.
