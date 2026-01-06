#!/usr/bin/env python3
import os
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "src"))

import speechbrain as sb  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402

from parkinsons_speech.utils import random_crop  # noqa: E402


class ParkinsonBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        feats = self.modules.feature_extractor(wavs)
        embeddings = self.modules.xvector(feats, lens)
        logits = self.modules.classifier(embeddings)
        predictions = self.hparams.log_softmax(logits)
        return predictions, lens

    def compute_objectives(self, predictions, batch, stage):
        preds, lens = predictions
        labels, _ = batch.label_encoded
        loss = self.hparams.compute_cost(preds, labels, lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, preds, labels, lens)
        return loss

    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        error = self.error_metrics.summarize("average")
        stats = {"loss": stage_loss, "error_rate": error, "accuracy": 1 - error}

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prep(hparams):
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label", "label_encoded")
    def label_pipeline(label):
        yield label
        yield label_encoder.encode_label_torch(label)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(
            sig,
            orig_freq=hparams["orig_sample_rate"],
            new_freq=hparams["sample_rate"],
        )
        sig = random_crop(sig, hparams["sample_rate"], hparams["chunk_duration"])
        max_val = torch.clamp(sig.abs().max(), min=1e-6)
        sig = sig / max_val
        return sig

    data_json = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    datasets = {}
    for name, path in data_json.items():
        datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=path,
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "label_encoded"],
        )

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[datasets["train"]], output_key="label"
    )
    label_encoder.expect_len(hparams["n_classes"])
    return datasets


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = dataio_prep(hparams)

    xvector_brain = ParkinsonBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    train_loader_opts = dict(hparams["dataloader_options"])
    valid_loader_opts = deepcopy(train_loader_opts)
    valid_loader_opts["shuffle"] = False

    xvector_brain.fit(
        epoch_counter=xvector_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=train_loader_opts,
        valid_loader_kwargs=valid_loader_opts,
    )

    xvector_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=valid_loader_opts,
    )
