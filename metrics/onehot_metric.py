import sys
sys.path.append('..')

from metrics import Metric
import torch

class OneHotMetric(Metric):
    def __init__(self):
        super(Metric, self).__init__()

    def reset(self):
        self.correct = 0
        self.total = 0
    
    def __call__(self, output, target):
        if output.dim() == 2:
            output = torch.argmax(output, dim=1)
        
        comp = output == target
        if type(comp) == bool:
            correct = 1 if comp else 0
        else:
            correct = comp.sum().item()
            
        self.correct += correct
        self.total += target.size(0)

        return correct / target.size(0)

    def __str__(self):
        return "Accuracy: %.2f%%" % (self.correct / self.total * 100)

    def get_log_title(self):
        return "Accuracy"

    def log(self):
        return str(self.correct / self.total)