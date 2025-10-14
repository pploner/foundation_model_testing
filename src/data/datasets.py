from torch.utils.data import IterableDataset
import numpy as np
import os
import random
import torch

class LocalVectorDataset(IterableDataset):
    """
    Iterates over locally stored .npy vectorized files in a structure like:
      base_dir/
          QCD_HT50toInf/
            file1_x.npy, file1_y.npy, ...
        base_dir: str, the base directory where the class subdirectories are located
        per_class_limit: optional int, how many samples to load per class (None for all)
    """
    def __init__(self, base_dir, per_class_limit=None):
        super().__init__()
        self.base_dir = base_dir
        self.per_class_limit = per_class_limit

    def __iter__(self):
        seed = torch.initial_seed() % 2**32
        
        class_dirs = [
            os.path.join(self.base_dir, folder)
            for folder in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, folder))
        ]

        rng = random.Random(seed)
        rng.shuffle(class_dirs)

        counters = {os.path.basename(c): 0 for c in class_dirs}

        for class_dir in class_dirs:
            cname = os.path.basename(class_dir)
            files_x = sorted(f for f in os.listdir(class_dir) if f.endswith("_x.npy"))
            rng.shuffle(files_x)

            for fx in files_x:
                fy = fx.replace("_x.npy", "_y.npy")
                X = np.load(os.path.join(class_dir, fx))
                y = np.load(os.path.join(class_dir, fy))
                
                X = torch.from_numpy(X).float()
                y = torch.from_numpy(y).long()

                if self.per_class_limit:
                    remain = self.per_class_limit - counters[cname]
                    if remain <= 0:
                        break
                    X = X[:remain]
                    y = y[:remain]
                    counters[cname] += len(y)
                else:
                    counters[cname] += len(y)

                yield X, y
                
class ShuffleBuffer(IterableDataset):
    def __init__(self, dataset, buffer_size=10000):
        """
        buffer_size: how many whole batches to buffer
        """
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        seed = torch.initial_seed() % 2**32
        rng = random.Random(seed)
        buf = []
        for batch in self.dataset:
            buf.append(batch)
            if len(buf) >= self.buffer_size:
                idx = rng.randrange(len(buf))
                yield buf.pop(idx)
        while buf:
            idx = rng.randrange(len(buf))
            yield buf.pop(idx)