# Model Card: Parkinson's Speech Classifiers

## Model Details
- **Developed by:** SpeechBrain recipes refactored from course notebook
- **Models:** Xvector, ECAPA-TDNN, wav2vec2-base, WavLM-base+, HuBERT-base
- **Task:** Binary classification of Parkinson's vs. non-Parkinson's speakers
- **Input:** 16 kHz mono wav, cropped/padded to 20 seconds

## Intended Use
- **Primary:** Research and coursework demonstrations on the Italian Parkinson Voice & Speech dataset.
- **Out-of-scope:** Any clinical or diagnostic decision making; deployment on individuals outside the dataset distribution.

## Data
- **Source:** [Italian Parkinson's Voice and Speech](https://dx.doi.org/10.21227/aw6b-tg17).
- **Labels:** `parkinson`, `not_parkinson` inferred from directory names.
- **Splitting:** Deterministic manifests with speaker-level default split (`scripts/prepare_manifests.py --split_by speaker`).

## Training Procedure
- **Framework:** SpeechBrain + PyTorch.
- **Hyperparameters:** See recipe-specific `hparams/train.yaml`.
- **Augmentation:** Optional additive noise and speed perturbation toggled in SSL recipes.
- **Checkpoints:** Best validation error saved under `results/<model>/<seed>/save/`.

## Evaluation
- **Metrics:** Error rate and accuracy logged per epoch; add F1/precision/recall in reports when available.
- **Repro:** `python scripts/prepare_manifests.py ...` then `python recipes/parkinsons_binary/<model>/train.py recipes/.../hparams/train.yaml --data_folder <raw_data_root>`.

## Limitations & Risks
- Small, imbalanced dataset; results may not generalize.
- Label inference relies on folder names; mis-filed samples will propagate errors.
- Models fine-tuned from English SSL checkpoints on Italian speech; accent/domain mismatch likely.

## Ethical Considerations
- Do not use for diagnosis or screening.
- Avoid using predictions without human oversight.
- If publishing results, report speaker-level splits and data preprocessing steps.
