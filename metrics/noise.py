from .metric import Metric
import statistics

class NoiseMetric(Metric):
    def __init__(self):
        super(NoiseMetric, self).__init__()

    def reset(self):
        self.noise_log = []

    def __call__(self, noise):
        self.noise_log.append(noise)

    def __str__(self):        
        noise_max = max(self.noise_log)
        noise_min = min(self.noise_log)
        noise_mean = statistics.mean(self.noise_log)
        noise_median = statistics.median(self.noise_log)

        return f"noise_max: {noise_max}, noise_min: {noise_min}, noise_mean: {noise_mean}, noise_median: {noise_median}"

    def get_log_title(self):
        return "noise_max,noise_min,noise_mean,noise_median"

    def log(self):
        noise_max = max(self.noise_log)
        noise_min = min(self.noise_log)
        noise_mean = statistics.mean(self.noise_log)
        noise_median = statistics.median(self.noise_log)

        return f"{noise_max},{noise_min},{noise_mean},{noise_median}"