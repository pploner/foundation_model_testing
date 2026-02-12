from typing import Any, Callable, Dict, Optional, Tuple

import os
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, ROC
from torchmetrics.classification import MulticlassAUROC, F1Score
from torchmetrics.classification.accuracy import Accuracy

from .components.transformer import TinyTransformer


class COLLIDE2VTransformerLitModule(LightningModule):
    """Example of a `LightningModule` for COLLIDE2V classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        optimizer: Callable,
        scheduler: Optional[Callable],
        compile: bool,
    ) -> None:
        """Initialize a `COLLIDE2VTransformerLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.normalizer = None

        self.net = None

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_auroc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (x, y).

        :return: (loss, probs, preds, targets)
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return loss, probs, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_auroc(probs, targets)
        self.val_roc.update(probs, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

        # TPR@FPR 1e-2 calculation and logging
        try:
            fpr, tpr, _ = self.val_roc.compute()   # shapes: (C, N)

            # TPR@FPR per class
            target_fpr = 1e-2
            tpr_at_fpr_vals = []

            for c in range(len(fpr)):
                fpr_c = fpr[c].detach().cpu().numpy()
                tpr_c = tpr[c].detach().cpu().numpy()

                tpr_interp = np.interp(target_fpr, fpr_c, tpr_c)
                tpr_at_fpr_vals.append(tpr_interp)

            tpr_macro = float(np.mean(tpr_at_fpr_vals))
            self.log("val/tpr_macro_at_1e-2", tpr_macro)

        except Exception as e:
            print(e)
            self.log("val/tpr_macro_at_1e-2", torch.tensor(0.0), prog_bar=False)

        self.val_roc.reset()

        return {"probs": probs}

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # Compute metrics only if AUROC has received any samples
        try:
            acc = self.val_acc.compute()
        except Exception:
            acc = torch.tensor(0.0, device=self.device)

        try:
            auroc = self.val_auroc.compute()
        except Exception:
            auroc = torch.tensor(0.0, device=self.device)

        self.val_acc_best(acc)  # update best so far val acc
        self.val_auroc_best(auroc)  # update best so far val auroc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=True)



    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"probs": probs}

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        if self.trainer and getattr(self.trainer, "datamodule", None):
            dm = self.trainer.datamodule

            num_classes = getattr(dm, "num_classes", None)

            eos_preproc_dir = getattr(dm, "paths", None)['eos_preproc_dir']
            feature_map_path = os.path.join(eos_preproc_dir, "feature_map.json")
            with open(feature_map_path, "r") as f:
                feature_map = json.load(f)

            self.normalizer = nn.Identity()

            # metric objects for calculating and averaging accuracy across batches
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            self.val_roc = ROC(task="multiclass", num_classes=num_classes)
            self.val_auroc = MulticlassAUROC(num_classes=num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

            self.net = TinyTransformer(
                feature_map=feature_map,
                d_model=self.hparams.d_model,
                n_heads=self.hparams.n_heads,
                num_layers=self.hparams.num_layers,
                d_ff=self.hparams.d_ff,
                dropout=self.hparams.dropout,
                num_classes=num_classes,
            )

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = COLLIDE2VTransformerLitModule(None, None, None, None)
