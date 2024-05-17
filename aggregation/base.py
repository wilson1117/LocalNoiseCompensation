import sys
sys.path.append("..")

from abc import ABC, abstractmethod
import torch
from metrics import get_metric, get_logs_title, get_logs, dump_logs, LossMetric, NoiseMetric
from tqdm import tqdm, trange
import random
import os
import gc


class Aggregator(ABC):
    def __init__(self, central_model, data_config, optimizer, logger, public_loader, args,
                 defender=None, metric=None, criterion=None, val_loader=None, test_loader=None, device=None):
        
        self.loss_metric = LossMetric()
        
        self.central_model = central_model
        self.data_config = data_config
        self.public_data = public_loader
        self.optimizer = optimizer(self.central_model.parameters())
        self.logger = logger
        self.args = args
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.noise_metric = NoiseMetric() if args.calc_noise else None
        self.defender = defender(args, self.noise_metric) if defender is not None else None


        self.metric = get_metric(data_config['metric'])() if metric is None else get_metric(metric)()
        self.criterion = getattr(torch.nn, data_config['criterion'])() if criterion is None else getattr(torch.nn, criterion)()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.model_folder = logger.create_folder("models")

    @staticmethod
    def run(self):
        pass

    @staticmethod
    def aggregate(self):
        pass

    def eval(self, dataloader, model=None):
        if model is None:
            model = self.central_model

        model.eval()
        self.loss_metric.reset()
        self.metric.reset()

        with torch.no_grad():
            for feature, labels in tqdm(dataloader):
                feature = feature.to(self.device)
                labels = labels.to(self.device)

                outputs = model(feature)
                loss = self.criterion(outputs, labels)

                self.loss_metric(loss.item(), len(labels))
                self.metric(outputs, labels)

        dump_logs(self.loss_metric, self.metric)
        
        return get_logs(self.loss_metric, self.metric)

