import torch

from ..utils import Model
from .pose_resnet import get_pose_model


def load_mdl(device=None, transform=None, mdl_dict={}, **kwargs):
    mdl = get_pose_model(**kwargs)
    return Model(mdl, device=device, transform=transform, **mdl_dict)

