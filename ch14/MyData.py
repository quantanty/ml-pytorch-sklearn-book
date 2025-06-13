import torch
import torchvision
from torchvision import transforms
from typing import Union
from torch.utils.data import Subset
import random

def load_transform():

    transform_train = transforms.Compose([
        transforms.RandomCrop((178, 178)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    transform = transforms.Compose([
        transforms.RandomCrop((178, 178)),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    return transform_train, transform

def load_CelebA(path:str='../data/', train: Union[bool, int] = True, valid: Union[bool, int]=True, test:bool=True, target_attr=18, random_seed=None):
    result = []

    transform_train, transform = load_transform()
    target_transform = lambda attr: attr[18]

    if train:
        train_dataset = torchvision.datasets.CelebA(path, split="train", target_type="attr", transform=transform_train, target_transform=target_transform, download=False)
        if isinstance(train, (int, float)) and not isinstance(train, bool):
            assert train > 0 and train <= len(train_dataset)
            positive = train // 2
            negative = train - positive
            positive_indices = torch.nonzero((train_dataset.attr[:, 18] == 1)).squeeze()
            positive_indices = positive_indices[:min(positive, len(positive_indices))]

            negative_indices = torch.nonzero((train_dataset.attr[:, 18] == 0)).squeeze()
            negative_indices = negative_indices[:min(negative, len(negative_indices))]

            indices = torch.cat([positive_indices, negative_indices])
            if random_seed:
                torch.manual_seed(random_seed)
            indices_ = torch.randperm(indices.size(0))
            indices = indices[indices_]

            train_dataset = Subset(train_dataset, indices)
        
        result.append(train_dataset)

    if valid:
        valid_dataset = torchvision.datasets.CelebA(path, split="valid", target_type="attr", transform=transform, target_transform=target_transform, download=False)
        if isinstance(valid, (int, float)) and not isinstance(valid, bool):
            assert valid > 0 and valid <= len(valid_dataset)
            valid_dataset = Subset(valid_dataset, torch.arange(valid))
        result.append(valid_dataset)

    if test:
        test_dataset = torchvision.datasets.CelebA(path, split='test', target_type='attr', transform=transform, target_transform=target_transform, download=False)
        result.append(test_dataset)

    if len(result) == 1:
        return result[0]
    return result