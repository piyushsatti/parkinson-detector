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

from parkinsons_speech.utils import prepare_label_encoder, random_crop  # noqa: E402


class ParkinsonBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.ssl_model(wavs, lens)
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.output_mlp(outputs)
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        labels, _ = batch.label_encoded
        labels = labels.squeeze(1)
        loss = self.hparams.compute_cost(predictions, labels)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.label, predictions, labels)
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
            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            old_lr_ssl, new_lr_ssl = self.hparams.lr_annealing_ssl(
                stats["error_rate"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.ssl_optimizer, new_lr_ssl
            )
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "ssl_lr": old_lr_ssl},
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
        torch.cuda.empty_cache()

    def init_optimizers(self):
        self.ssl_optimizer = self.hparams.ssl_opt_class(
            self.modules.ssl_model.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("ssl_opt", self.ssl_optimizer)
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "ssl_optimizer": self.ssl_optimizer,
        }


def dataio_prep(hparams):
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

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

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label", "label_encoded")
    def label_pipeline(label):
        yield label
        label_encoded = label_encoder.encode_label_torch(label)
        yield label_encoded

    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    datasets = {}
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["label", "sig", "label_encoded"],
        )

    label_encoder = prepare_label_encoder(
        datasets, hparams["save_folder"], output_key="label", expected_len=hparams["out_n_neurons"]
    )
    return datasets


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    if str(run_opts.get("device", "")).startswith("cuda") and not torch.cuda.is_available():
        run_opts["device"] = "cpu"

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = dataio_prep(hparams)

    hparams["ssl_model"] = hparams["ssl_model"].to(device=run_opts["device"])
    if not hparams["freeze_ssl"] and hparams["freeze_ssl_conv"]:
        hparams["ssl_model"].model.feature_extractor._freeze_parameters()

    language_brain = ParkinsonBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    train_loader_opts = dict(hparams["dataloader_options"])
    valid_loader_opts = deepcopy(train_loader_opts)
    valid_loader_opts["shuffle"] = False

    language_brain.fit(
        epoch_counter=language_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=train_loader_opts,
        valid_loader_kwargs=valid_loader_opts,
    )

    language_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=valid_loader_opts,
    )
