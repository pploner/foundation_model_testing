from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from lightning import Callback
from lightning.pytorch.loggers import Logger, MLFlowLogger


class SaveLogits(Callback):
    """
    Collects all test logits (N, C) and true labels (N,)
    and saves them into a single compressed .npz file.
    Also logs the file as an MLflow artifact.
    """

    def __init__(
        self,
        include_classnames: bool = True,
    ):
        super().__init__()
        self.artifact_dir = "predictions"
        self.filename = "test_logits_and_labels.npz"
        self.include_classnames = include_classnames

        self._logits: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    def on_test_epoch_start(self, trainer, pl_module):
        self._logits.clear()
        self._labels.clear()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if outputs is None or "logits" not in outputs:
            print("Warning: SaveLogits callback expected 'logits' in the outputs of the test step, but it was not found. Skipping this batch.")
            return

        logits = outputs["logits"]
        _, labels = batch

        self._logits.append(logits.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        if len(self._logits) == 0:
            print("[SaveTestLogitsNPZ] No logits collected. Did test_step return {'logits': logits}?")
            return

        logits = torch.cat(self._logits, dim=0).float().numpy()
        labels = torch.cat(self._labels, dim=0).numpy().astype(np.int64)

        payload = {
            "logits": logits,   # shape (N, C)
            "labels": labels,   # shape (N,)
        }

        if self.include_classnames and getattr(trainer, "datamodule", None) is not None:
            class_names = getattr(trainer.datamodule, "classnames", None)
            if class_names is not None:
                payload["class_names"] = np.array(class_names)

        out_dir = Path(trainer.default_root_dir) / "test_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.filename

        np.savez_compressed(out_path, **payload)

        # log to MLflow
        mlflow_logger = self.get_mlflow_logger(trainer)
        if mlflow_logger is None:
            print("[SaveTestLogitsNPZ] No MLFlowLogger found on trainer. Skipping artifact logging.")
            return

        mlflow_logger.experiment.log_artifact(
            run_id=mlflow_logger.run_id,
            local_path=str(out_path),
            artifact_path=self.artifact_dir,
        )
        print(f"[SaveTestLogitsNPZ] Logged MLflow artifact to '{self.artifact_dir}/'")

    def get_mlflow_logger(self, trainer) -> MLFlowLogger:
        """Extract the MLFlow logger from the trainer, if it is being used."""
        logger = trainer.logger
        if isinstance(logger, MLFlowLogger):
            return logger

        if isinstance(logger, Logger):
            return None

        for logger in getattr(trainer, "loggers", []) or []:
            if isinstance(logger, MLFlowLogger):
                return logger

        return None
