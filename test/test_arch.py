import torch
from motion.detect_human import HumanBBox

torch.manual_seed(42)


def test_arch():
    a = torch.rand(4, 3, 224, 224)
    mdl = HumanBBox(ap_sz=16)
    assert mdl(a).shape == (4, 4)

    mdl = HumanBBox(ap_sz=8)
    a = torch.rand(4, 3, 500, 500)
    assert mdl(a).shape == (4, 4)
