# Callback that computes the anomaly rate during training.
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from lightning.pytorch.loggers import MLFlowLogger, Logger
from lightning import Callback

from torchmetrics.classification import MulticlassAUROC
from torchmetrics.classification import MulticlassROC


class MCROC(Callback):
    """."""

    def __init__(self):
        super().__init__()
        self.device = None
        self.num_classes = None

    def on_validation_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""
        self.device = pl_module.device
        self.num_classes = trainer.datamodule.num_classes
        self.class_names = trainer.datamodule.classnames

    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.mcauc = MulticlassAUROC(num_classes=self.num_classes, average=None).to(self.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Determine the rate for every given metric for every validation data set."""
        self.total_batches = trainer.num_val_batches
        _, labels = batch
        self.mcauc.update(outputs['probs'], labels)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        aurocs = self.mcauc.compute()
        for idx, auroc in enumerate(aurocs):
            classname = self.class_names[idx]
            pl_module.log_dict(
                {f"val/class_{classname}_auc": auroc.item()},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        mean_auc = aurocs.mean()
        pl_module.log_dict(
            {f"val/mean_auc": mean_auc},
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def on_test_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""
        self.device = pl_module.device
        self.num_classes = trainer.datamodule.num_classes
        self.class_names = trainer.datamodule.classnames

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.mcroc = MulticlassROC(num_classes=self.num_classes, average=None).to(self.device)
        self.mcauc = MulticlassAUROC(num_classes=self.num_classes, average=None).to(self.device)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Determine the rate for every given metric for every test data set."""
        _, labels = batch
        self.mcauc.update(outputs['probs'], labels)
        self.mcroc.update(outputs['probs'], labels)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        fprs, tprs, thres = self.mcroc.compute()
        aurocs = self.mcauc.compute()
        for idx, auroc in enumerate(aurocs):
            classname = self.class_names[idx]
            pl_module.log_dict(
                {f"test/class_{classname}_auc": auroc.item()},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        mean_auc = aurocs.mean()
        pl_module.log_dict(
            {f"test/mean_auc": mean_auc},
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self._plot_rocs(trainer, fprs, tprs)

    def _plot_rocs(self, trainer, fprs, tprs):
        """plotting script."""
        plot_folder = Path(trainer.default_root_dir) / "roc_plots/"
        plot_folder.mkdir(parents=True, exist_ok=True)
        for c in range(len(fprs)):
            plt.figure()
            plt.plot(fprs[c].cpu(), tprs[c].cpu())
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC – Class {self.class_names[c]}")
            plt.grid(True)

            out_path = plot_folder / f"class_{self.class_names[c]}_roc.png"
            plt.savefig(out_path)
            plt.close()

        plt.figure(figsize=(8, 6))

        for c in range(len(fprs)):
            plt.plot(
                fprs[c].cpu(),
                tprs[c].cpu(),
                label=self.class_names[c],
                linewidth=1.5,
            )

        # optional random baseline
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (All Classes)")
        plt.legend(fontsize="small", ncol=2)
        plt.grid(True)

        combined_path = plot_folder / "roc_all_classes.png"
        plt.savefig(combined_path, bbox_inches="tight")
        plt.close()

        self.log_plots_to_mlflow(trainer, plot_folder)

    def log_plots_to_mlflow(
        self,
        trainer,
        plot_folder: Path,
    ):
        """Logs the plots generated by this callback to MLFlow."""
        mlflow_logger = self.get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        arti_dir = "rocs"

        # Log each image in the given plot_folder as an artifact.
        self.log_raw_imgs_to_mlflow(plot_folder, mlflow_logger, arti_dir)

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

    def log_raw_imgs_to_mlflow(self, plot_folder: Path, logger: MLFlowLogger, arti: Path):
        """Logs a directory of images to mlflow, in the artifact directory at arti."""
        IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        for img_path in sorted(plot_folder.glob("*")):
            if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
                logger.experiment.log_artifact(
                    run_id=logger.run_id,
                    local_path=str(img_path),
                    artifact_path=str(arti),
                )
