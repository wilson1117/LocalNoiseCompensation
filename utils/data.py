from torchvision import datasets
from torch.utils.data import Subset
import json
import os
import torch
from torch.utils.data import DataLoader
import numpy
import random

from .config import get_config

def load_data_config(path: str, rootdir: str, n: int = None):
    data_config = json.load(open(path))
    dataset_config = get_config(data_config['dataset'])

    params = dict(
        root=os.path.join(rootdir, data_config['dataset']),
        train=data_config['train'],
        download=True,
        transform=dataset_config['feature_transforms']
    )

    if 'split' in dataset_config:
        params['split'] = dataset_config['split']

    dataset = getattr(datasets, data_config['dataset'])(**params)
    
    public_datasets = Subset(dataset, data_config['public_indices'])
    client_datasets = [Subset(dataset, indices if n is None else indices[:n]) for indices in data_config['client_indices']]

    return (public_datasets, client_datasets), dataset_config


def get_dataset(name: str, rootdir: str, train: bool):
    config = get_config(name)

    params = dict(
        root=os.path.join(rootdir, name),
        train=train,
        download=True,
        transform=config['feature_transforms']
    )

    if 'split' in config:
        params['split'] = config['split']

    dataset = getattr(datasets, name)(**params)

    return dataset, config

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def to_dataloader(datasets, batch_size, shuffle=True, num_workers=0, seed=None):

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        if type(datasets) is list:
            return [DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=seed_worker, generator=g) for dataset in datasets]
        
        return DataLoader(datasets, batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    else:
        if type(datasets) is list:
            return [DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers) for dataset in datasets]
        
        return DataLoader(datasets, batch_size, shuffle=shuffle, num_workers=num_workers)