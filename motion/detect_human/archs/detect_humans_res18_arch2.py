import torch
import torch.nn as nn
from torchvision import models

from ..fastai_utils import AdaptiveConcatPool2d, Flatten, bn_drop_lin


class HumanBBox(nn.Module):
    def __init__(self, ap_sz=32, p=0.5, num_outpur=4, arch=None):
        super(HumanBBox, self).__init__()

        if arch is None:
            arch = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(arch.children())[:2])
        self.in_between = nn.Sequential(*[AdaptiveConcatPool2d((ap_sz, ap_sz))])
        self.cnn2 = nn.Sequential(
            *[
                nn.Conv2d(128, 64, 1),
                nn.Conv2d(64, 32, 1),
                nn.Conv2d(32, 4, 1),
                Flatten(),
            ]
        )
        head = bn_drop_lin(4096, 512, p=p / 2, actn=torch.nn.ReLU(inplace=True))
        head += bn_drop_lin(512, num_outpur, p=p)
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.cnn(x)
        x = self.in_between(x)
        x = self.cnn2(x)
        x = self.head(x)

        return x.sigmoid_()

