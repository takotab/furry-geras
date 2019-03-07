import cv2
import imutils
import torch

from motion.orient_mdl import load_mdl
from motion.detect_human_ssd import PredictionTransform

f = "test/assets/boy_beer.jpg"

device = torch.device("cpu")

# as it turns out my mdl is not good enough
# and does not reconize the rotated (270) boy
def test_orientmdl():
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trans = PredictionTransform(300)

    mdl = load_mdl(device=device)
    for angle_i in range(4):
        angle = [0, 90, 180, 270][angle_i]
        _im = imutils.rotate_bound(img, angle)
        _im = trans(_im)[None, :].float().to(device)
        pred = mdl(_im).detach().numpy().argmax()
        assert pred == angle_i

