# not yet done
import torch
import torch.nn as nn
from torchvision import models

# from fastai.callbacks.hooks import num_features_model
# from fastai.vision import create_head
from ..fastai_utils import *


class HumanBBox(nn.Module):
    def __init__(self, num_outpur=4, arch=None):
        super(HumanBBox, self).__init__()

        if arch is None:
            arch = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(arch.children())[:2])
        self.in_between = nn.Sequential(*[AdaptiveConcatPool2d((8, 8)), Flatten()])

        head = bn_drop_lin(8192, 512, p=0.25, actn=torch.nn.ReLU(inplace=True))
        head += bn_drop_lin(512, num_outpur, p=0.5)
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.cnn(x)
        x = self.in_between(x)
        x = self.head(x)

        return x.sigmoid_()


if __name__ == "__main__":
    a = torch.rand(4, 3, 224, 224)
    mdl = HumanBBox(4)
    assert mdl(a).shape == (4, 4)

    a = torch.rand(4, 3, 500, 500)
    assert mdl(a).shape == (4, 4)

