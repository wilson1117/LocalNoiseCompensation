from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_log_title():
        pass

    @abstractmethod
    def log(self):
        pass

def dump_logs(*logs):
    print(", ".join([str(log) for log in logs]))

def get_logs_title(*logs):
    return ",".join([log.get_log_title() for log in logs])

def get_logs(*logs):
    return ",".join([log.log() for log in logs])