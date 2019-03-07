import torch
from motion.orient_mdl import OrientMdl


def test_arch_orient():
    mdl = OrientMdl()
    mdl.eval()
    a = torch.rand(4, 3, 300, 300)
    assert mdl(a).detach().cpu().numpy().shape == (4, 4)
