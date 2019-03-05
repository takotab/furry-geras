import cv2
import numpy as np
import torch

from motion.detect_human_ssd import load_mdl
from motion.detect_human_ssd import predict_image, predict_video
from motion import get_video_array
from motion import make_video

f = "test/assets/boy_beer.jpg"
size = 393, 640
results = [274.0, 118.0, 401.0, 434.0]


def load_detect_human_mdl(**kwargs):
    mdl = load_mdl(**kwargs)

    a = torch.rand(1, 3, 300, 300)
    mdl_lst = mdl.net(a)
    print([o.shape for o in mdl_lst])
    return mdl


def test_human_loc():
    mdl = load_detect_human_mdl()
    # vid_array = get_video_array(f)
    # size = [vid_array.shape[s] for s in (1, 2, 1, 2)]
    boxes, labels, probs = predict_image(f, mdl, save_result=True)
    print(boxes, labels, probs, boxes[0])
    _boy_beer_check(boxes)
    orig_image = cv2.imread(f)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)[None, :]
    video_array = np.concatenate([image] * 17, 0)
    lst = predict_video(video_array, mdl)
    print([o[0].shape for o in lst], "*", len(lst[0]))
    _boy_beer_check(lst[0][0])
    # assert np.mean(np.abs(bbox - results)) < 0.04


def _boy_beer_check(boxes):
    boxes = boxes[0].detach().numpy()
    assert np.round(boxes[0]) == 353
    assert np.round(boxes[1]) == 65
    assert np.round(boxes[2]) == 487
    assert np.round(boxes[3]) == 385
