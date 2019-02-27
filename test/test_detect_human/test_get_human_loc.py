import torch

from motion.detect_human import load_mdl
from motion.detect_human import get_human_loc
from motion import get_video_array

f = "/home/tako/devtools/furry-geras/test/data/000000050638.jpg"
size = 393, 640
results = [344.43, 65.35, 127.94, 316.04999999999995]


def load_detect_human_mdl():
    mdl = load_mdl()

    a = torch.rand(4, 3, 500, 500)
    assert mdl(a).shape == (4, 4)
    return mdl


def test_human_loc():
    mdl = load_detect_human_mdl()
    vid_array = get_video_array(f)
    size = [vid_array.shape[s] for s in (1, 2, 1, 2)]
    bbox = get_human_loc(mdl, vid_array)
    print(bbox * size)
    assert 1 == 0

