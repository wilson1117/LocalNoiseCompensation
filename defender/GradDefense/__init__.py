import sys
sys.path.append("..")

from defender.base import Defender
from .sensitivity import compute_sens

class GradDefense(Defender):
    def __init__(self, args, noise_metric=None, for_attacker=False):
        super(GradDefense, self).__init__(args, noise_metric, for_attacker)

        if args.gd_clip:
            from .clip import noise
        else:
            from .perturb import noise

        self.noise = noise

        self.slices_num = args.gd_slices_num
        self.perturb_slices_num = args.gd_perturb_slices_num
        self.scale = args.gd_scale

    def prepare(self, model, public_dataset, loss_fn, device):
        self.device = device

        model = model.to(device)
        self.sensitivity = compute_sens(model, public_dataset, device, loss_fn)
        model = model.cpu()

    def local_gradient_defense(self, grad, model, epoch, batch, prev_info=None):
        grad = [g.cpu() for g in grad]

        grad = self.noise(
            dy_dx=[grad,],
            sensitivity=self.sensitivity,
            slices_num=self.slices_num,
            perturb_slices_num=self.perturb_slices_num,
            noise_intensity=self.scale,
        )

        return [g.to(self.device) for g in grad], None