import argparse
from functools import partial
import torch
from torch import optim
import os

import aggregation
from model import get_classification_model
from utils import *
import defender

RESULT_DIR = "results"
DATASET_ROOT_DIR = "dataset"

# torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # Dataset and Model Config
    argparser.add_argument('--data-config', type=str, default='data_config/OutPost_CIFAR10_100_1.0_123.json', help='data config file')
    argparser.add_argument('--model', type=str, default='LeNet', help='model name')


    # Federated Learning Config
    argparser.add_argument('--algorithm', type=str, default='FedAvg', help='aggregation algorithm name')
    argparser.add_argument('--num-rounds', type=int, default=500, help='number of rounds')
    argparser.add_argument('--random-select', type=int, default=10, help='number of clients selected in each round')

    # FedAvg Config
    argparser.add_argument('--local-epochs', type=int, default=5, help='number of local epochs')


    # Learning Hyperparameters
    argparser.add_argument('--optimizer', type=str, default='SGD', help='optimizer name')
    argparser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    argparser.add_argument('--batch-size', type=int, default=32, help='batch size')

    # SGD Config
    argparser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    argparser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')


    # Defender Config
    argparser.add_argument('--defender', type=str, default=None, help='defender name')
    argparser.add_argument('--calc-noise', action='store_true', help='calculate noise')

    # LocalCompensation Config
    argparser.add_argument('--delta', type=float, default=1, help='max noise scale')
    argparser.add_argument('--disable-compensation', action='store_true', help='disable compensation')
    argparser.add_argument('--clip', type=float, default=None, help='clip upperbound')

    # OutPost Config
    argparser.add_argument('--noise-base', type=float, default=0.8, help='noise base')
    argparser.add_argument('--phi', type=float, default=40, help='phi')
    argparser.add_argument('--prune-base', type=float, default=80, help='prune base')
    argparser.add_argument('--beta', type=float, default=0.1, help='beta')

    # GradDefense Config
    argparser.add_argument('--gd-clip', action='store_true', help='clip noise')
    argparser.add_argument('--gd-slices-num', type=int, default=10, help='slices num')
    argparser.add_argument('--gd-perturb-slices-num', type=int, default=5, help='perturb slices num')
    argparser.add_argument('--gd-scale', type=float, default=0.01, help='scale')

    # # ANP Config
    argparser.add_argument('--anp-delta', type=float, default=1e-7)
    argparser.add_argument('--anp-epsilon', type=float, default=3e-3)
    

    # Metric Config
    argparser.add_argument('--criterion', type=str, default=None, help='criterion')
    argparser.add_argument('--metric', type=str, default=None, help='metric')

    # Other Config
    argparser.add_argument('--test-batch', type=int, default=256, help='test batch size')
    argparser.add_argument('--seed', type=int, default=123, help='random seed')
    argparser.add_argument('--log-dir', type=str, default=None, help='log directory')
    argparser.add_argument('--device', type=str, default=None, help='device')
    argparser.add_argument('--save-freq', type=int, default=10, help='save frequency')

    args = argparser.parse_args()

    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load data config
    (public_datasets, client_datasets), dataset_config = data.load_data_config(args.data_config, DATASET_ROOT_DIR)
    train_loaders = data.to_dataloader(client_datasets, args.batch_size, shuffle=True, seed=args.seed)
    pub_loader = data.to_dataloader(public_datasets, len(public_datasets), shuffle=False)

    test_dataset, test_dataset_config = data.get_dataset(dataset_config['name'], DATASET_ROOT_DIR, train=False)
    test_loader = data.to_dataloader(test_dataset, args.test_batch, shuffle=False)

    # Load model
    if dataset_config['type'] == 'classification':
        model = get_classification_model(args.model, dataset_config['num_classes'], dataset_config['label_type'], dataset_config['input_shape'], dataset_config['grayscale'])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")
    
    # Create optimizer
    if args.optimizer == "SGD":
        optimizer = partial(optim.SGD, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = partial(optim.Adam, lr=args.lr)

    # Load defender
    if args.defender is not None:
        defender_cls = getattr(defender, args.defender)
    else:
        defender_cls = None
    
    # Create Logger
    if args.log_dir is not None:
        logger_obj = logger.Logger(os.path.join(RESULT_DIR, args.log_dir))
    else:
        logger_obj = logger.Logger(os.path.join(RESULT_DIR, f"{args.algorithm}_{args.model}_{dataset_config['name']}_{args.defender if args.defender is not None else 'Origin'}_{len(train_loaders)}_{args.batch_size}"))

    logger_obj.create_log("params.json").write_json(vars(args))

    # Create Aggregation Algorithm
    agg_algorithm = getattr(aggregation, args.algorithm)(model, dataset_config, optimizer, logger_obj, pub_loader, args,
                                                         defender=defender_cls, metric=args.metric, criterion=args.criterion,
                                                         test_loader=test_loader, device=args.device)
    
    # Run
    agg_algorithm.run(train_loaders, args.num_rounds, args.random_select, args.save_freq)