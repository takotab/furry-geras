import numpy as np
import torch

import motion
from motion import BBoxPreds


def test_crop_human():
    array = np.random.rand(4, 600, 600, 3)
    bbox_results = BBoxPreds(
        torch.Tensor([[410, 60, 220, 150]]), torch.Tensor([15]), torch.Tensor([0.95])
    )
    size = (200, 250)
    im = motion.crop_to_human(array, [bbox_results] * 4, size)

    assert list(im.shape) == [4, *size[::-1], 3]

    # non-human
    bbox_results.set(0, "label", 2)
    bbox_results._incl_human()
    try:
        im = motion.crop_to_human(array, [bbox_results] * 4)
        assert False
    except NotImplementedError:
        assert True

