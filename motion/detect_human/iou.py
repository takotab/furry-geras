import torch
from torch import nn


class IoU(nn.modules.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, preds, targs):
        return _IoU(preds, targs)


def intersection(preds, targs):
    # preds and targs are of shape (bs, 4), pascal_voc format
    max_xy = torch.min(preds[:, 2:], targs[:, 2:])
    min_xy = torch.max(preds[:, :2], targs[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def union(preds, targs):
    return area(preds) + area(targs) - intersection(preds, targs)


def _IoU(preds, targs):
    return intersection(preds, targs) / union(preds, targs)
