import cv2
import imutils
import torch

from motion.orient_mdl import load_mdl
from motion.orient_mdl import rotate

f = "test/assets/boy_beer.jpg"

device = torch.device("cpu")


def test_orientmdl(angle_i=None):
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mdl = load_mdl(device=device)
    if angle_i is None:
        angle_i = 3

    angle = [0, 90, 180, 270][angle_i]
    _im = imutils.rotate_bound(img, angle)
    print(img.shape, _im.shape)
    rotated_vid = rotate([_im] * 7, mdl)
    print([o.shape for o in rotated_vid], img.shape)
    assert rotated_vid[0].shape == img.shape
    assert rotated_vid[0][0, 0, 0] == img[0, 0, 0]

