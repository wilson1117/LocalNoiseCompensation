from .metric import Metric, get_logs_title, get_logs, dump_logs
from .onehot_metric import OneHotMetric
from .loss_metric import LossMetric
from .img_metric import ImageMetric
from .time import TimeMetric
from .noise import NoiseMetric
import torch

__all__ = ['Metric', 'OneHotMetric', 'LossMetric', 'TimeMetric', 'get_logs_title', 'get_logs', 'dump_logs', 'ImageMetric', 'NoiseMetric', 'total_variation']

metric_list = {
    'OneHotMetric': OneHotMetric,
    'LossMetric': LossMetric,
    'ImageMetric': ImageMetric
}

def get_metric(name):
    return metric_list[name]

def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
