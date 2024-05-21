from .base import Attacker
from functools import partial
import torch

class IG(Attacker):
    def __init__(self, model, loss_fn, device, args, dataset_config, defender=None):
        self.init = 'randn'
        self.attack_optim = 'Adam'
        self.attack_lr = 0.1
        self.attack_steps = 3000
        self.attack_dist = 'sim'
        self.tv = 1e-4

        self.tv = args.tv if args.tv is not None else self.tv
        
        super(IG, self).__init__(model, loss_fn, device, args, dataset_config, defender)

        self.scheduler = partial(
            torch.optim.lr_scheduler.MultiStepLR,
            milestones=[
                self.attack_steps // 2.667,
                self.attack_steps // 1.6,
                self.attack_steps // 1.142,
            ],
            gamma=0.1
        )
    
    def imp_optim(self, dummy_feature, dummy_label):
        return self.match_optimizer([dummy_feature])
    
    def init_dummy_data(self, feature_shape, target_label):
        dummy_feature = self.init_tensor(feature_shape, self.device)
        return dummy_feature, target_label.to(self.device)
    
    def auxiliary_loss(self, dummy_feature, dummy_label):
        return self.tv * self.total_variation(dummy_feature)

    @staticmethod
    def total_variation(x):
        """Anisotropic TV."""
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy