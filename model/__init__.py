from .lenet import LeNet
from .ConvNet import ConvNet
from .CNN import CNN
from .resnet import *
from .modules import PatchedModule
from torchvision import models as torch_models
from torch import nn
import re

model_list = {
    "LeNet": LeNet,
    "ConvNet": ConvNet,
    "CNN": CNN,
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet1202": resnet1202
}

def get_model_class(name):
    name = re.sub("resnet", "ResNet", name)

    return name

def get_classification_model(name, num_classes, label_type, input_shape, grayscale, pretrain=None, attack_test=False):
    if name in model_list:
        model = model_list[name](num_classes=num_classes, attack_test=attack_test, input_shape=input_shape, grayscale=grayscale)
    elif hasattr(torch_models, name):
        if label_type == "onehot":
            if pretrain is not None:
                pretrain_weight = getattr(getattr(torch_models, f"{get_model_class(name)}_Weights"), pretrain, None)
            else:
                pretrain_weight = None

            if pretrain_weight is not None:
                print(f"Use pretrain: {pretrain}")

            model = getattr(torch_models, name)(weights=pretrain_weight)
    else:
        raise ValueError(f"Model {name} not found.")
        
    return model