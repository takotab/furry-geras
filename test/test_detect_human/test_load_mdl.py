import torch

from motion.detect_human import load_mdl


def test_load_mdl():
    mdl = load_mdl()

    a = torch.rand(4, 3, 500, 500)
    assert mdl(a).shape == (4, 4)

