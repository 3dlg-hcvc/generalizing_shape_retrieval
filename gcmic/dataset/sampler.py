# https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py

from typing import Callable, Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import Sampler, RandomSampler, BatchSampler
import torchvision


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class UniqueRandomSampler(Sampler):
    def __init__(self, data_source: Sized, labels: np.ndarray = None, batch_size: int = None, num_samples: Optional[int] = None, generator = None) -> None:
        self.data_source = data_source
        self.labels = labels
        self.batch_size = batch_size
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
            
        all_rand_idx = torch.randperm(n, generator=generator).numpy()
        all_rand_labels = self.labels[all_rand_idx]
        num_batches = self.num_samples // self.batch_size if self.num_samples % self.batch_size == 0 else (self.num_samples + self.batch_size - 1) // self.batch_size
        
        for b in range(num_batches):
            batched_rand_idx = all_rand_idx[b*self.batch_size:(b+1)*self.batch_size]
            batched_rand_labels = all_rand_labels[b*self.batch_size:(b+1)*self.batch_size]
            batched_unique_labels, inv_indices, counts = np.unique(batched_rand_labels, return_inverse=True, return_counts=True)
            
            if len(batched_unique_labels) != self.batch_size:
                idx_candidates = np.nonzero(~np.in1d(self.labels, batched_unique_labels))[0]
                repeated_labels_unique_indices = np.nonzero(counts > 1)[0]
                for i in repeated_labels_unique_indices:
                    repeated_labels = batched_unique_labels[i]
                    repeated_labels_indices = np.nonzero(inv_indices == i)[0][1:]
                    batched_rand_idx[repeated_labels_indices] = np.random.choice(idx_candidates, len(repeated_labels_indices))
        
        yield from all_rand_idx.tolist()

        # for _ in range(self.num_samples // 32):
        #     yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
        # yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        
        # yield from torch.randperm(n, generator=generator).tolist()

    def __len__(self) -> int:
        return self.num_samples
    