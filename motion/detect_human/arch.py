# not yet done
import torch
import torch.nn as nn
from torchvision import models


class HumanBBox(nn.module):
    def __init__(self, num_outpur, arch=None):
        if arch is None:
            arch = models.resnet50()
        mdl = [arch]

        mdl.append(bn_drop_lin(arch))
        nn.Sequential()


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0.0, actn=None):
    "FASTAI: Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers
