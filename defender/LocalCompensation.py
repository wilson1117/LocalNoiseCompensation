from .base import Defender
import torch
from torch import nn
import numpy as np

class LocalCompensation(Defender):
    def __init__(self, args, noise_metric=None, for_attacker=False):
        super(LocalCompensation, self).__init__(args, noise_metric, for_attacker)

        self.disable_compensation = args.disable_compensation
        self.delta = args.delta
        self.clip = args.clip

    def local_gradient_defense(self, grad, model, epoch, batch, prev_info=None):
        return self.share_gradient_defense(grad, model, prev_info)

    def share_gradient_defense(self, grad, model, prev_info=None):
        grad_len = len(grad)

        if prev_info is None:
            prev_info = [None] * grad_len

        # if self.clip:
        #     for idx, g in enumerate(grad):
        #         grad[idx] /= max(1, g.norm(2) / self.clip)

        next_info = None if self.for_attacker or self.disable_compensation else [None] * grad_len
        
        for (idx, g), prev_noise in zip(enumerate(grad), prev_info):
            noise_mag = (g.detach().abs().max() * self.delta).item()

            if self.noise_metric is not None:
                self.noise_metric(noise_mag)

            # skip nan
            if noise_mag != noise_mag:
                continue

            noise = torch.normal(mean=0, std=noise_mag, size=g.size(), device=g.device)

            if self.for_attacker or self.disable_compensation:
                grad[idx] = g + noise
            else:
                if prev_noise is None:
                    grad[idx] = g + noise
                else:
                    grad[idx] = g + noise - prev_noise
                
                next_info[idx] = noise

        del prev_info

        return grad, next_info
    
    def attack_simulation(self, grad):
        if self.enable_clip:
            for idx, g in enumerate(grad):
                norm = g.norm(2).item()
                if norm > 1:
                    grad[idx] /= norm
        
        return grad