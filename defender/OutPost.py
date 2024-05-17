from .base import Defender
import torch
from torch import nn
import numpy as np
import random

class OutPost(Defender):
    def __init__(self, args, noise_metric=None, for_attacker=False):
        super(OutPost, self).__init__(args, noise_metric, for_attacker)

        self.noise_base = args.noise_base
        self.phi = args.phi
        self.prune_base = args.prune_base
        self.beta = args.beta

    def local_gradient_defense(self, grad, model, epoch, batch, prev_info=None):
        iteration = epoch * (batch + 1)
        # Probability decay
        if random.random() < 1 / (1 + self.beta * iteration):
            risk = self.compute_risk(model)
            grad = self.noise(grad, risk)

        elif self.noise_metric is not None:
            for _ in range(len(grad)):
                self.noise_metric(0)


        return grad, prev_info
    
    def share_gradient_defense(self, grad, model, prev_info=None):
        risk = self.compute_risk(model)
        grad = self.noise(grad, risk)

        return grad, prev_info

    @staticmethod
    def compute_risk(model: nn.Module):
        var = []
        for param in model.parameters():
            var.append(torch.var(param).cpu().detach().numpy())
        var = [min(v, 1) for v in var]
        return var


    def noise(self, dy_dx: list, risk: list):
        # Calculate empirical FIM
        fim = []
        flattened_fim = None
        for i in range(len(dy_dx)):
            squared_grad = dy_dx[i].clone().pow(2).mean(0).cpu().detach().numpy()
            fim.append(squared_grad)
            if flattened_fim is None:
                flattened_fim = squared_grad.flatten()
            else:
                flattened_fim = np.append(flattened_fim, squared_grad.flatten())

        fim_thresh = np.percentile(flattened_fim, 100 - self.phi)

        for i in range(len(dy_dx)):
            # pruning
            grad_tensor = dy_dx[i].cpu().detach().numpy()
            flattened_weights = np.abs(grad_tensor.flatten())
            thresh = np.percentile(flattened_weights, self.prune_base)
            grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
            # noise
            if self.noise_metric is not None:
                self.noise_metric(risk[i] * self.noise_base)
            noise_base = torch.normal(
                0, risk[i] * self.noise_base, dy_dx[i].shape
            )
            noise_mask = np.where(fim[i] < fim_thresh, 0, 1)
            gauss_noise = noise_base * noise_mask
            dy_dx[i] = (
                (torch.Tensor(grad_tensor) + gauss_noise)
                .to(dtype=torch.float32)
                .to(dy_dx[i].device)
            )

        return dy_dx