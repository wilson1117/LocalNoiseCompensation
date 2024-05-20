from .base import Defender
import math
import torch

class ANP(Defender):
    def __init__(self, args, for_attacker=False):
        super(ANP, self).__init__(args, for_attacker)

        self.epsilon = args.anp_epsilon
        self.delta = args.anp_delta
        self.clip = args.clip
        self.warmup = args.warmup

    def share_weight_defense(self, origin_state, state, model, round, num_sample, prev_info=None):
        # max_param = max([param.detach().norm().item() for param in model.parameters() if param.requires_grad])
        # sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * (2 * min(max_param, self.clip) * 0.01 / num_sample) / self.epsilon
        # sigma /= 100
        sigma = 0.003
        # print(sigma)

        for param in model.parameters():
                param.data = param / max(1, param.norm() / self.clip)

        if round < self.warmup:
            for param in model.parameters():
                noise = torch.normal(mean=0, std=sigma, size=param.size(), device=param.device)
                param.data = param + noise
        else:
            param_flatten = torch.cat([param.flatten() for param in model.parameters() if param.requires_grad])
            noise = torch.normal(mean=0, std=sigma, size=param_flatten.size(), device=param_flatten.device)

            param_sorted, I = torch.sort(param_flatten)

            if round % 2 == 1:
                noise_desc = torch.sort(noise, descending=True)[0]
                param_sorted = param_sorted + noise_desc
            else:
                noise_asc = torch.sort(noise)[0]
                param_sorted = param_sorted + noise_asc

            # param_flatten = param_sorted[I]
            param_flatten.scatter_(0, I, param_sorted)
            idx = 0
            for param in model.parameters():
                if not param.requires_grad:
                    continue

                if param.requires_grad:
                    param.data = param_flatten[idx:idx+param.numel()].view(param.size())
                    idx += param.numel()

        # for param in model.parameters():
        #     param.data = param / max(1, param.norm() / self.clip)
        #     noise = torch.normal(mean=0, std=sigma, size=param.size(), device=param.device)

        #     if round <= self.warmup:
        #         param.data = param + noise
        #     else:
        #         noise = noise.flatten()
        #         param_flatten = param.flatten()

        #         param_sorted = torch.sort(param_flatten)[0]
        #         I = torch.argsort(param_flatten)

        #         if round % 2 == 1:
        #             noise_desc = torch.sort(noise, descending=True)[0]
        #             param_sorted = param_sorted + noise_desc
        #         else:
        #             noise_asc = torch.sort(noise)[0]
        #             param_sorted = param_sorted + noise_asc
                
        #         param.data = param_sorted[I].view(param.size())

        return model.state_dict(), None