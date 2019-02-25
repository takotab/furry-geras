import torch
from motion.detect_human import HumanBBox

torch.manual_seed(42)


def test_arch():
    a = torch.rand(4, 3, 224, 224)
    mdl = HumanBBox(4)
    assert mdl(a).shape == (4, 4)
