from abc import ABC, abstractmethod
from .patch import PatchedModule
from functools import partial
from collections import OrderedDict
from torch.nn import functional as F
import math
import torch

class Attacker(ABC):
    def __init__(self, model, loss_fn, device, args, dataset_config, defender=None):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.args = args
        self.dataset_config = dataset_config
        self.defender = defender
        self.scheduler = None


        self.calc_delta = args.calc_delta
        self.local_epoch = args.local_epoch
        self.batch_size = args.batch_size
        self.lr = args.lr

        self.init = args.init if args.init is not None else self.init
        self.attack_optim = args.attack_optim if args.attack_optim is not None else self.attack_optim
        self.attack_lr = args.attack_lr if args.attack_lr is not None else self.attack_lr
        self.attack_steps = args.attack_steps if args.attack_steps is not None else self.attack_steps
        self.attack_dist = args.attack_dist if args.attack_dist is not None else self.attack_dist

        if self.attack_optim == "Adam":
            self.match_optimizer = partial(torch.optim.Adam, lr=self.attack_lr)
        elif self.attack_optim == "SGD":
            self.match_optimizer = partial(torch.optim.SGD, lr=self.attack_lr, momentum=0.9, nesterov=True)
        elif self.attack_optim == "LBFGS":
            self.match_optimizer = partial(torch.optim.LBFGS, lr=self.attack_lr)


    def attack(self, origin_state, share_weights, feature_shape, target_label):
        self.origin_state = origin_state
        self.model.load_state_dict(origin_state)

        target_weight = [w.to(self.device) for w in share_weights]

        if self.calc_delta:
            origin_weight = [w.to(self.device) for w in self.model.parameters()]
            target_delta = [t_weight - o_weight for o_weight, t_weight in zip(origin_weight, target_weight)]
        else:
            origin_weight = None
            target_delta = None

        dummy_feature, dummy_label = self.init_dummy_data(feature_shape, target_label)
        optimizer = self.imp_optim(dummy_feature, dummy_label)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)

        closure = partial(self.closure, origin_weight, target_weight, target_delta, optimizer, dummy_feature, dummy_label)

        for round in range(self.attack_steps):
            loss = optimizer.step(closure)
            if self.scheduler is not None:
                scheduler.step()

            if loss.isnan():
                print("early stop")
                print("loss:", loss.item())
                break

            if round == 0 or (round + 1) % 100 == 0:
                print(f"Round {round + 1}:", loss.item())

        return dummy_feature.detach(), dummy_label.detach()
    
    def closure(self, origin_weight, target_weight, target_delta, optimizer, dummy_feature, dummy_label):
        optimizer.zero_grad()

        self.model.load_state_dict(self.origin_state)        
        patched_model = PatchedModule(self.model)

        for _ in range(self.local_epoch):
            for idx in range(int(math.ceil(dummy_feature.shape[0] / self.batch_size))):
                outputs = patched_model(
                    dummy_feature[idx * self.batch_size : (idx + 1) * self.batch_size],
                    patched_model.parameters
                )

                if dummy_label.dtype == torch.float and isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    # CrossEntropy for onehot
                    loss = torch.mean(torch.sum(- F.softmax(dummy_label[idx * self.batch_size : (idx + 1) * self.batch_size], dim=-1) * F.log_softmax(outputs, dim=-1), 1))
                else:
                    loss = self.loss_fn(outputs, dummy_label[idx * self.batch_size : (idx + 1) * self.batch_size])

                grad = torch.autograd.grad(
                    loss,
                    patched_model.parameters.values(),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                )

                if self.defender is not None:
                    grad = self.defender.attack_process(grad)

                patched_model.parameters = OrderedDict(
                    (name, param - self.lr * grad_part)
                    for ((name, param), grad_part) in zip(
                        patched_model.parameters.items(), grad
                    )
                )
        
        dummy_weight = patched_model.parameters.values()

        if self.calc_delta:
            dummy_delta = [d_weight - o_weight for o_weight, d_weight in zip(origin_weight, dummy_weight)]
            rec_loss = self.calc_dist(target_delta, dummy_delta) + self.auxiliary_loss(dummy_feature, dummy_label)
        else:
            rec_loss = self.calc_dist(target_weight, dummy_weight) + self.auxiliary_loss(dummy_feature, dummy_label)

        rec_loss.backward()

        return rec_loss

    def imp_optim(self, dummy_feature, dummy_label):
        return self.match_optimizer([dummy_feature, dummy_label])
    
    def auxiliary_loss(self, dummy_feature, dummy_label):
        return 0

    @abstractmethod
    def init_dummy_data(self, feature_shape, target_label):
        raise NotImplementedError()
    
    def init_tensor(self, shape, device):
        if self.init == "randn":
            return torch.randn(shape).to(device).requires_grad_(True)
        elif self.init == "rand":
            return ((torch.rand(shape) - 0.5) * 2).to(device).requires_grad_(True)
        elif self.init == "zeros":
            return torch.zeros(shape).to(device).requires_grad_(True)
        elif self.init == "half":
            return (torch.ones(shape) - 0.5).to(device).requires_grad_(True)
        
    def calc_dist(self, target_weight, dummy_weight):
        pnorm = [0, 0]
        rec_loss = 0
        
        for share_weight, dummy_weight in zip(target_weight, dummy_weight):
            if self.attack_dist == "l2":
                rec_loss += ((dummy_weight - share_weight).pow(2)).sum()
            elif self.attack_dist == "l1":
                rec_loss += ((dummy_weight - share_weight).abs()).sum()
            elif self.attack_dist == "max":
                rec_loss += ((dummy_weight - share_weight).abs()).max()
            elif self.attack_dist == "sim":
                rec_loss -= (dummy_weight * share_weight).sum()
                pnorm[0] += dummy_weight.pow(2).sum()
                pnorm[1] += share_weight.pow(2).sum()
            elif self.attack_dist == "simlocal":
                rec_loss += (
                    1
                    - torch.nn.functional.cosine_similarity(
                        dummy_weight.flatten(), share_weight.flatten(), 0, 1e-10
                    )
                )
            
        if self.attack_dist == "sim":
            rec_loss = 1 + rec_loss / pnorm[0].sqrt() / pnorm[1].sqrt()


        return rec_loss