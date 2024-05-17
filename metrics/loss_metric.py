from .metric import Metric

class LossMetric(Metric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.loss = 0
        self.total = 0

    def __call__(self, loss, batch_size):
        self.loss += loss * batch_size
        self.total += batch_size

    def __str__(self):
        return "Loss: %.4f" % (self.loss / self.total)

    def get_log_title(self):
        return "Loss"

    def log(self):
        return str(self.loss / self.total)