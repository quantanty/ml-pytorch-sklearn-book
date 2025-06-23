import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import math

__all__ = [
    "MysteriousIsland",
    "split_dataset",
]

class MysteriousIsland(Dataset):
    def __init__(self, path, chunk_size, start=None, end=None, keep_text=True, keep_text_encoded=True):
        super().__init__()
        with open(path, 'r', encoding='utf8') as fp:
            text = fp.read()

        if start and end:
            start_idx = text.find(start)
            end_idx = text.find(end)
            text = text[start_idx:end_idx]

        chars = sorted(set(text))
        self.char2int = {ch:i for i, ch in enumerate(chars)}
        self.int2char = np.array(chars)

        text_encoded = np.array([self.char2int[ch] for ch in text])
        self.chunks = torch.tensor(np.array([text_encoded[i:i+chunk_size] for i in range(len(text_encoded) - chunk_size)]))

        if keep_text:
            self.text = text
        
        if keep_text_encoded:
            self.text_encoded = text_encoded

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1].long(), chunk[1:].long()
    
    def get_encoder_decoder(self):
        encoder = lambda text: np.array([self.char2int[ch] for ch in text])
        decoder = lambda seq: ''.join(self.int2char[seq])
        return encoder, decoder
    
def split_dataset(dataset, train_size=None, test_size=None, random_state=None):
    
    n_samples = len(dataset)

    if test_size is None and train_size is None:
        test_size = 0.25
    
    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (
        test_size_type == "i"
        and (test_size >= n_samples or test_size <= 0)
        or test_size_type == "f"
        and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            "test_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(test_size, n_samples)
        )

    if (
        train_size_type == "i"
        and (train_size >= n_samples or train_size <= 0)
        or train_size_type == "f"
        and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            "train_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(train_size, n_samples)
        )
    
    if train_size is not None and train_size_type not in ("i", "f"):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError("Invalid value for test_size: {}".format(test_size))
    
    if train_size_type == "f" and test_size_type == "f" and train_size + test_size > 1:
        raise ValueError(
            "The sum of test_size and train_size = {}, should be in the (0, 1)"
            " range. Reduce test_size and/or train_size.".format(train_size + test_size)
        )
    
    if test_size_type == "f":
        n_test = math.ceil(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    if train_size_type == "f":
        n_train = math.floor(train_size * n_samples)
    elif train_size_type == "i":
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    indices = torch.randperm(len(dataset))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:n_train+n_test]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)