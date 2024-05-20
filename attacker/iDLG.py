from .base import Attacker
import torch

class iDLG(Attacker):
    def __init__(self, model, loss_fn, device, args, dataset_config, defender=None):
        self.init = 'randn'
        self.attack_optim = 'LBFGS'
        self.attack_lr = 1
        self.attack_steps = 5000
        self.attack_dist = 'l2'
        
        super(iDLG, self).__init__(model, loss_fn, device, args, dataset_config, defender)
    
    def imp_optim(self, dummy_feature, dummy_label):
        return self.match_optimizer([dummy_feature])
    
    def init_dummy_data(self, feature_shape, target_label):
        dummy_feature = self.init_tensor(feature_shape, self.device)
        return dummy_feature, target_label.to(self.device)