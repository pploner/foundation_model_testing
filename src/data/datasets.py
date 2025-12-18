import os
import random
import numpy as np
import torch
import math
import json
from torch.utils.data import IterableDataset


class LocalVectorDataset(IterableDataset):
    """Stream samples from preprocessed .npy shards on EOS or local storage."""

    def __init__(self, base_dir, per_class_limit=None, shuffle_file_order=True, classnames=None, folder_map=None):
        super().__init__()
        self.base_dir = base_dir
        self.per_class_limit = per_class_limit
        self.shuffle = shuffle_file_order

        # Either find all class directories or use provided class names and folder map
        if classnames is None:
            class_dirs = [
                os.path.join(self.base_dir, folder)
                for folder in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, folder))
            ]
        else:
            folder_names = []
            for cname in classnames:
                if folder_map and cname in folder_map:
                    folder_names.append(folder_map[cname])
                else:
                    raise ValueError(f"Class name {cname} not found in folder_map.")
            class_dirs = [os.path.join(self.base_dir, d) for d in folder_names]
            for d in class_dirs:
                if not os.path.isdir(d):
                    raise ValueError(f"Provided folder name {d} is not a directory in base_dir {self.base_dir}")

        self.class_dirs = class_dirs

        all_files = []
        for class_dir in self.class_dirs:
            files_x = [f for f in os.listdir(class_dir) if f.endswith("_x.npy")]
            for fx in files_x:
                all_files.append((class_dir, fx))

        self.all_files = all_files

        # --- Global shuffle across all class shards ---
        if self.shuffle:
            rng = random.Random()
            rng.shuffle(self.all_files)

    def __iter__(self):
        # --- Worker setup ---
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        # --- Split non-overlapping chunks for each worker ---
        per_worker = int(math.ceil(len(self.all_files) / float(num_workers)))
        start = worker_id * per_worker
        end = min(start + per_worker, len(self.all_files))
        my_files = self.all_files[start:end]

        counters = {}

        if self.per_class_limit is not None:
            worker_class_limit = int(math.ceil(self.per_class_limit / float(num_workers)))
        else:
            worker_class_limit = None

        # --- Iterate files ---
        for class_dir, fx in my_files:
            cname = os.path.basename(class_dir)
            fy = fx.replace("_x.npy", "_y.npy")

            # Memory-map (efficient lazy loading)
            X = np.load(os.path.join(class_dir, fx), mmap_mode="r")
            y = np.load(os.path.join(class_dir, fy), mmap_mode="r")

            n = len(y)
            counters.setdefault(cname, 0)
            remain = None if worker_class_limit is None else worker_class_limit - counters[cname]
            if remain is not None and remain <= 0:
                continue

            limit = n if remain is None else min(remain, n)

            for i in range(limit):
                yield torch.from_numpy(X[i].copy()).float(), torch.tensor(y[i]).long()

            counters[cname] += limit



class ShuffleBuffer(IterableDataset):
    """Shuffle streaming samples from an IterableDataset."""

    def __init__(self, dataset, buffer_size=20000):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        seed = torch.initial_seed() % 2**32
        rng = random.Random(seed)
        buf = []

        for sample in self.dataset:
            buf.append(sample)
            if len(buf) >= self.buffer_size:
                idx = rng.randrange(len(buf))
                yield buf.pop(idx)

        while buf:
            idx = rng.randrange(len(buf))
            yield buf.pop(idx)
