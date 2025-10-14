from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import os, random, numpy as np

from src.data.utils import get_all_cols, compute_vlen, vectorized_to_local, worker_init_fn, has_enough_events, estimate_mean_std

from src.data.datasets import LocalVectorDataset, ShuffleBuffer     

class COLLIDE2VDataModule(LightningDataModule):
    """`LightningDataModule` for the COLLIDE2V dataset.

    The COLLIDE2V dataset is a dataset for training and evaluating particle collision models, simulated by a Madgraph Pythia Delphes chain.
    It is stored in columnar format in Parquet files, where the rows correspond to collision events and the columns represent the physics features of the particles involved in the collisions.
    The dataset is organized in folders based on the process type and stored in shards of 10'000 simulated events each.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        batch_size: int = 1024,
        train_val_test_split_per_class: Tuple[int, int, int] = (500_000, 10_000, 10_000),
        num_workers: int = 0,
        pin_memory: bool = False,
        label: str = "test",
        paths: Optional[Dict[str, str]] = None,
        datasets_config: Optional[Dict[str, Any]] = None,
        to_classify: Optional[Dict[str, str]] = None,
        process_to_folder: Optional[Dict[str, str]] = None,
        seed: int = 42,
    ):
        """Initialize a `COLLIDE2VDataModule`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.train_val_test_split_per_class = train_val_test_split_per_class
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.label = label
        self.paths = paths or {}
        self.datasets_config = datasets_config or {}
        self.to_classify = to_classify or {}
        self.process_to_folder = process_to_folder or {}
        
        self.vlen = compute_vlen(self.datasets_config)

        self.classnames = list(self.to_classify.keys())
        self.pretty = {c: self.to_classify[c] for c in self.classnames}
        self.folder = {c: self.process_to_folder[self.pretty[c]] for c in self.classnames}
        self.labels = {c: i for i, c in enumerate(self.classnames)}

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.seed = seed

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes .
        """
        return len(self.classnames)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        
        print(f"🟡 Generating vectorized data in {self.paths['eos_vec_dir']}")
        vectorized_to_local(
            base_dir=self.paths["dataset_dir"],
            config=self.datasets_config,
            class_names=self.classnames,
            folder_map=self.folder,
            labels_map=self.labels,
            all_cols=get_all_cols(self.datasets_config),
            vlen=self.vlen,
            afs_vec_dir=self.paths["afs_vec_dir"],
            eos_vec_dir=self.paths["eos_vec_dir"],
            split_counts=self.train_val_test_split_per_class,
            read_batch_size=512, 
        )

        mean, std = estimate_mean_std(os.path.join(self.paths["eos_vec_dir"], "train"), 30000)
        self.feature_mean = mean
        self.feature_std = std

        print(f"🟡 For first 10 features, estimated mean: {self.feature_mean[:10]}"
              f" and std: {self.feature_std[:10]}")

        # INCLUDE HERE PREPROCESSING STEPS AND PLOTTING IF NEEDED

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if has_enough_events(self.paths["eos_preproc_dir"]):
            print(f"🟡 Preprocessed data found — using from {self.paths['eos_preproc_dir']}")
            self.trainstream = LocalVectorDataset(os.path.join(self.paths["eos_preproc_dir"], "train"))
            self.valstream = LocalVectorDataset(os.path.join(self.paths["eos_preproc_dir"], "val"))
            self.teststream = LocalVectorDataset(os.path.join(self.paths["eos_preproc_dir"], "test"))
        else:
            print(f"🟡 Preprocessed data not found — using vectorized data from {self.paths['eos_vec_dir']}")
            self.trainstream = LocalVectorDataset(os.path.join(self.paths["eos_vec_dir"], "train"))
            self.valstream = LocalVectorDataset(os.path.join(self.paths["eos_vec_dir"], "val"))
            self.teststream = LocalVectorDataset(os.path.join(self.paths["eos_vec_dir"], "test"))

        self.shuffled_train = ShuffleBuffer(self.trainstream, buffer_size=10000)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(dataset=self.shuffled_train, batch_size=None, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory, worker_init_fn=worker_init_fn)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(dataset=self.valstream, batch_size=None, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory, worker_init_fn=worker_init_fn)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(dataset=self.teststream, batch_size=None, num_workers=self.num_workers, 
                          pin_memory=self.pin_memory, worker_init_fn=worker_init_fn)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = COLLIDE2VDataModule()
