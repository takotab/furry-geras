# not yet done
import torch
import torch.nn as nn
from torchvision import models
from fastai.callbacks.hooks import num_features_model
from fastai.vision import create_head


class HumanBBox(nn.Module):
    def __init__(self, num_outpur=4, arch=None):
        super(HumanBBox, self).__init__()

        if arch is None:
            arch = models.resnet50(pretrained=True)
        self.cnn = arch
        head = bn_drop_lin(1000, 512, p=0.25, actn=torch.nn.ReLU(inplace=True))
        head += bn_drop_lin(512, num_outpur, p=0.5)
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.cnn(x)
        x = self.head(x)

        return 1.02 * x.sigmoid_() - 0.01


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0.0, actn=None):
    "FASTAI: Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


# if __name__ == "__main__":
#     a = torch.rand(4, 3, 224, 224)
#     mdl = HumanBBox(4)
#     print(mdl(a))
