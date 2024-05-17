import sys
sys.path.append("..")

from .base import Aggregator
from copy import deepcopy
from tqdm import tqdm
import random
import torch
import os
import random

from metrics import get_logs, TimeMetric

class FedAvg(Aggregator):
    def __init__(self, central_model, data_config, optimizer, logger, public_loader, args,
                 defender=None, metric=None, criterion=None, val_loader=None, test_loader=None, device=None):
        
        super(FedAvg, self).__init__(central_model, data_config, optimizer, logger, public_loader, args,
                                     defender, metric, criterion, val_loader, test_loader, device)

        self.client_optimizer = optimizer
        self.local_epoch = args.local_epochs

    def run(self, client_loaders, rounds, random_select, save_freq):
        if self.test_loader is not None:
            train_logger = self.logger.create_log("train.csv")
            if self.noise_metric is not None:
                train_logger.log(
                    "Round,Test Loss,Test Acc,Execute Time,Cumulative Time," + 
                    self.noise_metric.get_log_title()
                )
            else:
                train_logger.log("Round,Test Loss,Test Acc,Execute Time,Cumulative Time")

        clients = [
            Client(self.central_model, loader, self.data_config, self.client_optimizer, self.criterion, self.public_data, self.defender)
            for loader in client_loaders
        ]

        if self.defender is not None:
            self.defender.prepare(self.central_model, self.public_data, self.criterion, self.device)

        time_metric = TimeMetric()
        
        for r in range(rounds):
            print(f"\n\nRound {r+1}/{rounds}")

            share_states = [None] * random_select

            origin_state = deepcopy(self.central_model.state_dict())

            if self.noise_metric is not None:
                self.noise_metric.reset()

            for idx, client in enumerate(tqdm(random.sample(clients, random_select))):
                share_states[idx] = client.train(origin_state, self.local_epoch, self.device)

                torch.cuda.empty_cache()

            self.aggregate(share_states)

            self.central_model.to(self.device)

            execute_time = time_metric()
            
            print("\nEvaluation:")

            test_logs = self.eval(self.test_loader)

            if self.noise_metric is not None:
                train_logger.log(f"{r + 1},{test_logs},{execute_time},{time_metric.log()},{self.noise_metric.log()}")
            else:
                train_logger.log(f"{r + 1},{test_logs},{execute_time},{time_metric.log()}")

            if r == 0 or (r + 1) % save_freq == 0:
                torch.save(self.central_model.state_dict(), os.path.join(self.model_folder, "Round_" + str(r + 1) + ".pth"))

            self.central_model.cpu()

    def aggregate(self, share_states):
        total_sample = sum([sample_count for _, sample_count in share_states])
        self.central_model.load_state_dict({
            key: sum(state[key] * (sample_count / total_sample) for state, sample_count in share_states)
            for key in share_states[0][0].keys() if "num_batches_tracked" not in key
        })

class Client:
    def __init__(self, model, dataloader, data_config, optimizer, loss_fn, public_data, defender=None):
        self.model = deepcopy(model)
        self.dataloader = dataloader
        self.optimizer = optimizer(self.model.parameters())
        self.loss_fn = loss_fn
        self.defender = defender
        self.public_data = public_data
        self.num_dataset = len(self.dataloader.dataset)

        self.prev_info = None

    def train(self, origin_state, epochs, device, output_weights=False, create_graph=False):
        self.model = self.model.to(device)
        self.model.train()
        self.model.load_state_dict(origin_state)

        

        for epoch in range(epochs):
            for batch, (feature, labels) in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                feature = feature.to(device)
                labels = labels.to(device)

                outputs = self.model(feature)
                loss = self.loss_fn(outputs, labels)

                grad = list(torch.autograd.grad(loss, self.model.parameters(), create_graph=create_graph))
                
                if self.defender is not None:
                    grad, self.prev_info = self.defender.local_gradient_defense(grad, self.model, epoch, batch, prev_info=self.prev_info)
                    
                for g, param in zip(grad, self.model.parameters()):
                    param.grad = g

                self.optimizer.step()
                if not create_graph:
                    self.optimizer.zero_grad()

        result_state = self.model.state_dict()

        if self.defender is not None:
            result_state, self.prev_info = self.defender.share_weight_defense(origin_state, result_state, self.model, self.num_dataset, prev_info=self.prev_info)
        
        self.model = self.model.cpu()
        
        if output_weights:
            return self.model.parameters(), len(self.dataloader.dataset)

        return result_state, len(self.dataloader.dataset)
        