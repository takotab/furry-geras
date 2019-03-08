import cv2
import torch
import motion
from motion import pose_resnet

f = "test/assets/divera_trend2.png"

target = [
    [53.0, 23.0],
    [53.0, 21.0],
    [53.0, 21.0],
    [42.0, 19.0],
    [50.0, 23.0],
    [32.0, 24.0],
    [48.0, 35.0],
    [22.0, 38.0],
    [52.0, 47.0],
    [19.0, 49.0],
    [43.0, 55.0],
    [26.0, 53.0],
    [38.0, 56.0],
    [26.0, 72.0],
    [36.0, 74.0],
    [27.0, 89.0],
    [36.0, 92.0],
]


def test_pose():
    img = cv2.imread(f)

    mdl = pose_resnet.get_pose_model(device=torch.device("cpu"))
    pred = pose_resnet.get_pose([img], mdl)
    # motion.plot_pose(img, pred[0], save=False)
    assert (target == pred[0]).all()
