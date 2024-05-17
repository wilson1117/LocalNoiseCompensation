import argparse
import torch
from torch.utils.data import WeightedRandomSampler
import random
from tqdm import tqdm
import numpy as np
import json
import os

from utils import data, set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset info
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--testset', dest='train', action='store_false', help='use testset')

    # Generate setting
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-clients', type=int, default=100, help='number of clients')
    parser.add_argument('--pubset-sample-per-class', type=int, default=10, help='public set sample per class')
    parser.add_argument('--partition-size', type=int, default=None, help='partition size')
    parser.add_argument('--concentration', type=float, default=1.0, help='concentration parameter')

    # Others setting
    parser.add_argument('--rootdir', type=str, default='dataset', help='root directory')
    parser.add_argument('--outdir', type=str, default='data_config', help='output directory')

    args = parser.parse_args()

    set_seed(args.seed)

    dataset, config = data.get_dataset(args.dataset, args.rootdir, args.train)
    num_classes = config['num_classes']

    if args.partition_size is None:
        args.partition_size = len(dataset) // args.num_clients
    
    print("create class map...")
    class_map = [label for _, label in tqdm(dataset)]
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    selected_pubset = [[] for _ in range(num_classes)]
    pubset = []
    
    for idx in indices:
        current_class = class_map[idx]
        if len(selected_pubset[current_class]) < args.pubset_sample_per_class:
            selected_pubset[current_class].append(idx)
            pubset.append(idx)
        elif len(pubset) == args.pubset_sample_per_class * num_classes:
            break

    client_indices = []

    for client_idx in range(args.num_clients):
        target_proportions = np.random.dirichlet(
            np.repeat(args.concentration, num_classes)
        )

        sample_weights = target_proportions[class_map]

        gen = torch.Generator()
        gen.manual_seed(args.seed + client_idx)

        indices = list(WeightedRandomSampler(
            weights=sample_weights,
            num_samples=args.partition_size,
            replacement=False,
            generator=gen,
        ))

        client_indices.append(indices)

    filename = f'OutPost_{args.dataset}_{args.num_clients}_{args.concentration}_{args.seed}.json'
    filepath = os.path.join(args.outdir, filename)

    os.makedirs(args.outdir, exist_ok=True)

    with open(filepath, 'w') as file:
        json.dump(dict(
            **vars(args),
            client_sample_count=[len(idcs) for idcs in client_indices],
            public_indices=pubset,
            client_indices=client_indices
        ), file)