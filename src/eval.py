from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import functools
import torch
import omegaconf
import collections
import typing
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

# Allow safe unpickling of functools.partial from trusted checkpoints (else newer Pytorch versions fail to load them)
torch.serialization.add_safe_globals([functools.partial,
                                      torch.optim.AdamW,torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.ReduceLROnPlateau,
                                      omegaconf.ListConfig, omegaconf.DictConfig, omegaconf.dictconfig.DictConfig,
                                      omegaconf.nodes.AnyNode, omegaconf.base.Metadata, omegaconf.base.ContainerMetadata,
                                      collections.defaultdict, typing.Any,
                                      list, dict, int])

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

def _find_hydra_config_for_ckpt(ckpt_path: str) -> Path | None:
    """Walk upwards from ckpt_path looking for <trial>/.hydra/config.yaml."""
    p = Path(ckpt_path).expanduser().resolve()
    for parent in [p.parent, *p.parents]:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.exists():
            return candidate
    return None

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    if cfg.get("use_trial_config", True):
        # keep ONLY these from eval.yaml
        eval_ckpt_path = cfg.ckpt_path
        eval_task_name = cfg.get("task_name", "eval")
        eval_mlflow_experiment_name = OmegaConf.select(cfg, "logger.mlflow.experiment_name")
        eval_mlflow_run_name = OmegaConf.select(cfg, "logger.mlflow.run_name")

        hydra_cfg_path = _find_hydra_config_for_ckpt(eval_ckpt_path)
        if hydra_cfg_path is None:
            raise FileNotFoundError(
                f"Could not find .hydra/config.yaml for ckpt_path={eval_ckpt_path}. "
                "Expected <trial>/.hydra/config.yaml in parent dirs."
            )

        train_cfg = OmegaConf.load(hydra_cfg_path)

        # make train_cfg editable
        OmegaConf.set_struct(train_cfg, False)

        # base config is the training run (so model/data params match checkpoint)
        cfg = train_cfg
        # but overwrite these with eval.yaml (e.g. logger for evaluation results, task_name for MLflow tags, etc.)
        OmegaConf.update(cfg, "ckpt_path", eval_ckpt_path, merge=False)
        OmegaConf.update(cfg, "task_name", eval_task_name, merge=False)
        OmegaConf.update(cfg, "logger.mlflow.experiment_name", eval_mlflow_experiment_name, merge=False)
        OmegaConf.update(cfg, "logger.mlflow.run_name", eval_mlflow_run_name, merge=False)
    evaluate(cfg)


if __name__ == "__main__":
    main()
