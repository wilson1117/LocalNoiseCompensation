from abc import ABC


class Defender(ABC):
    def __init__(self, args, noise_metric=None, for_attacker=False):
        self.args = args
        self.for_attacker = for_attacker
        self.noise_metric = noise_metric

    def prepare(self, model, public_dataset, loss_fn, device):
        pass


    # For share weight
    def local_gradient_defense(self, grad, model, epoch, batch, prev_info=None):
        return grad, prev_info

    def share_weight_defense(self, origin_state, state, model, num_sample, prev_info=None):
        return state, prev_info


    # For share gradient
    def share_gradient_defense(self, grad, model, prev_info=None):
        pass

    def attack_simulation(self, grad):
        return grad