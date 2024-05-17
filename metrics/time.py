from .metric import Metric
import time

class TimeMetric(Metric):
    def reset(self):
        self.end = self.start = time.time()

    def __call__(self):
        next_time = time.time()
        exec_time = next_time - self.end

        self.end = next_time
        return exec_time
    
    def __str__(self):
        return "Time: %.2f" % (self.end - self.start)
    
    def get_log_title(self):
        return "Time"
    
    def log(self):
        return str(self.end - self.start)