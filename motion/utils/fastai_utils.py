import torch
import torch.nn as nn


class Flatten(nn.Module):
    "FASTAI: Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"

    def __init__(self, full: bool = False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class ConcatPool2d(nn.Module):
    "Layer that concats `AvgPool2d` and `MaxPool2d`."

    def __init__(self, kernel_sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        kernel_sz = kernel_sz or 1
        self.ap,self.mp = nn.AvgPool2d(kernel_sz), nn.MaxPool2d(kernel_sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)



class AdaptiveConcatPool2d(nn.Module):
    "FASTAI: Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0.0, actn=None):
    "FASTAI: Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers
