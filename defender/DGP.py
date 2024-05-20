from .base import Defender
import torch
from torch import nn
import numpy as np

class DGP(Defender):
    def __init__(self, args, noise_metric=None, for_attacker=False):
        super(DGP, self).__init__(args, noise_metric, for_attacker)

        

    def local_gradient_defense(self, grad, model, epoch, batch, prev_info=None):

        for idx, g in enumerate(grad):
            upper_bound = torch.kthvalue()

        return grad, None