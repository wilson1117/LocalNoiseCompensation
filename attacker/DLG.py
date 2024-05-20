from .base import Attacker
import torch

class DLG(Attacker):
    def __init__(self, model, loss_fn, device, args, dataset_config, defender=None):
        self.init = 'randn'
        self.attack_optim = 'LBFGS'
        self.attack_lr = 1
        self.attack_steps = 5000
        self.attack_dist = 'l2'
        
        super(DLG, self).__init__(model, loss_fn, device, args, dataset_config, defender)

    def init_dummy_data(self, feature_shape, target_label):
        dummy_feature = self.init_tensor(feature_shape, self.device)
        if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
            dummy_label = self.init_tensor([*target_label.shape, self.dataset_config['num_classes']], self.device)
        else:
            dummy_label = self.init_tensor(target_label.shape, self.device)
        return dummy_feature, dummy_label
    
    def imp_optim(self, dummy_feature, dummy_label):
        return self.match_optimizer([dummy_feature, dummy_label])