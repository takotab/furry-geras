# not yet done
import torch
import torch.nn as nn
from torchvision import models

# from fastai.callbacks.hooks import num_features_model
# from fastai.vision import create_head
from .fastai_utils import *


class HumanBBox(nn.Module):
    def __init__(self, ap_sz=8, p=0.5, num_outpur=4, arch=None):
        super(HumanBBox, self).__init__()

        if arch is None:
            arch = models.resnet34(pretrained=True)
        self.cnn = nn.Sequential(*list(arch.children())[:2])
        self.in_between = nn.Sequential(
            *[AdaptiveConcatPool2d((ap_sz, ap_sz)), Flatten()]
        )
        # self.cnn2 = nn.Sequential(
        #     *[
        #         nn.Conv2d(128, 64, 1),
        #         nn.Conv2d(64, 32, 1),
        #         nn.Conv2d(32, 4, 1),
        #         nn.Conv2d(4, 1, 1),
        #         Flatten(),
        #     ]
        # )
        # self.in_between = nn.Sequential(*[AdaptiveConcatPool2d((32, 32))])
        # nn.Dropout2d(p, inplace=True),

        # self.num_channels = int(4096 / ((ap_sz ** 2) * 2))
        head = bn_drop_lin(4096, 512, p=p / 2, actn=torch.nn.ReLU(inplace=True))
        head += bn_drop_lin(512, num_outpur, p=p)
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.cnn(x)
        x = self.in_between(x[:, :32, :, :])
        # x = self.cnn2(x)
        x = self.head(x[:, :4096])

        return x.sigmoid_()


if __name__ == "__main__":
    a = torch.rand(4, 3, 224, 224)
    mdl = HumanBBox(4)
    assert mdl(a).shape == (4, 4)

    a = torch.rand(4, 3, 500, 500)
    assert mdl(a).shape == (4, 4)

