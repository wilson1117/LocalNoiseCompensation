import argparse
import attacker
import torch
from torch import optim
from functools import partial
import os
from copy import deepcopy

from utils import set_seed, data, logger, img
from model import get_classification_model
from metrics import AttackMetric
import defender

DATASET_ROOT_DIR = "dataset"
RESULT_DIR = "attack_result"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Dataset Config
    parser.add_argument('--data-config', type=str, default='data_config/OutPost_CIFAR10_100_1.0_123.json', help='data config file')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--sample-per-client' , type=int, default=1, help='number of samples per client')

    # Model Config
    parser.add_argument('--model', type=str, default='LeNet', help='model name')
    parser.add_argument('--model-state', type=str, default=None, help='model state file')

    # Aggregation Config
    parser.add_argument('--algorithm', type=str, default='FedAvg', help='aggregation algorithm name')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--criterion', type=str, default=None, help='criterion name')
    parser.add_argument('--local-epoch', type=int, default=1, help='number of local epochs')


    # Attacker Config
    parser.add_argument("--limit-round", type=int, default=10)
    parser.add_argument("--attacker", type=str, default="DLG")
    parser.add_argument("--use-weight", dest="calc_delta", action="store_false")
    parser.add_argument("--init", type=str, default=None)
    parser.add_argument("--attack-optim", type=str, default=None)
    parser.add_argument("--attack-lr", type=float, default=None)
    parser.add_argument("--attack-steps", type=int, default=None)
    parser.add_argument("--attack-dist", type=str, default=None)
    parser.add_argument("--tv", type=float, default=None)

    # Defender Config
    defender.get_args(parser)

    # Other Config
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--device', type=str, default=None, help='device')


    args = parser.parse_args()
    
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else args.device

    # Load Data Config
    (public_datasets, client_datasets), dataset_config = data.load_data_config(args.data_config, DATASET_ROOT_DIR, n=args.sample_per_client)
    train_loaders = data.to_dataloader(client_datasets, args.batch_size, shuffle=False, seed=args.seed)
    public_loaders = data.to_dataloader(public_datasets, len(public_datasets), shuffle=False, seed=args.seed)

    # Load Model
    if dataset_config['type'] == 'classification':
        model = get_classification_model(args.model, dataset_config['num_classes'], dataset_config['label_type'], dataset_config['input_shape'], dataset_config['grayscale'], attack_test=True)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")
    
    if args.model_state is not None:
        model.load_state_dict(torch.load(args.model_state))

    model.to(device)

    # Create optimizer
    if args.optimizer == "SGD":
        optimizer = partial(optim.SGD, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = partial(optim.Adam, lr=args.lr)

    # Load Criterion
    criterion = getattr(torch.nn, dataset_config['criterion'])() if args.criterion is None else getattr(torch.nn, args.criterion)()

    # Load Metric
    metric = AttackMetric()

    # Logger
    logger_dir = os.path.join(RESULT_DIR, f"{args.attacker}_{dataset_config['name']}_{args.model}_{args.defender if args.defender is not None else 'Origin'}_{args.seed}")
    print(f"Logger Directory: {logger_dir}")
    logger_obj = logger.Logger(logger_dir)

    logger_obj.create_log("params.json").write_json(vars(args))

    result_log = logger_obj.create_log("result.csv")
    result_log.log(f"Round,{metric.get_log_title()}")

    if args.defender is not None:
        defender_obj = getattr(defender, args.defender)(args)
    else:
        defender_obj = None

    attacker_obj = getattr(attacker, args.attacker)(model, criterion, device, args, dataset_config, defender_obj)

    attack_count = 0

    if args.algorithm == "FedAvg":
        from aggregation.FedAvg import Client

        origin_state = deepcopy(model.state_dict())

        if defender_obj is not None:
            defender_obj.prepare(model, public_loaders, criterion, device)

        clients = [
            (Client(model, loader, dataset_config, optimizer, criterion, public_datasets, defender=defender_obj), loader)
            for loader in train_loaders
        ]

        for client, dataloader in clients:
            if attack_count >= args.limit_round:
                break

            print(f"Round: {attack_count}")

            origin_feature = torch.cat(list(map(lambda data: data[0], dataloader)))
            origin_label = torch.cat(list(map(lambda data: data[1], dataloader)))

            share_weights = list(client.train(origin_state, args.local_epoch, 1, device, output_weights=True)[0])
            
            dummy_feature, dummy_label = attacker_obj.attack(origin_state, share_weights, origin_feature.shape, origin_label)

            origin_imgs = img.to_img(origin_feature.cpu(), dataset_config['normalize'][0], dataset_config['normalize'][1])
            dummy_imgs = img.to_img(dummy_feature.cpu(), dataset_config['normalize'][0], dataset_config['normalize'][1])


            img_folder = f"attack_imgs/attack_{attack_count}"
            img_folder = logger_obj.create_folder(img_folder)

            psnr, ssim, lpips = metric(dummy_imgs, origin_imgs)

            print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}\n")
            result_log.log(f"{attack_count},{metric.log()}")

            origin_imgs = img.swap_channel(origin_imgs)
            dummy_imgs = img.swap_channel(dummy_imgs)

            for idx in range(len(origin_imgs)):
                img.save_img(origin_imgs[idx], os.path.join(img_folder, f"origin_{idx}.png"))

            for idx in range(len(dummy_imgs)):
                img.save_img(dummy_imgs[idx], os.path.join(img_folder, f"dummy_{idx}.png"))

            img.save_img(origin_imgs, os.path.join(img_folder, "origin_imgs.png"))
            img.save_img(dummy_imgs, os.path.join(img_folder, "dummy_imgs.png"))

            img.save_img(torch._stack([dummy_imgs, origin_imgs]), os.path.join(img_folder, "compare_imgs.png"))

            attack_count += 1

        logger_obj.create_log("best.json").write_json(metric.output_best())



            