import torch
from motion.detect_human.archs.detect_humans_res18_arch2 import HumanBBox as c_HumanBBox
from motion.detect_human.arch import HumanBBox

torch.manual_seed(42)


def test_arch():
    mdl = c_HumanBBox(ap_sz=32)
    a = torch.rand(4, 3, 500, 500)
    assert mdl(a).shape == (4, 4)

    a = torch.rand(4, 3, 1920, 1080)
    assert mdl(a).shape == (4, 4)

    a = torch.rand(4, 3, 224, 224)
    mdl = HumanBBox(ap_sz=64)
    assert mdl(a).shape == (4, 4)
