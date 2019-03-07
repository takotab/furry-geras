import torch
import imutils

from motion.orient_mdl import load_mdl
from motion.detect_human_ssd import PredictionTransform

trans = PredictionTransform(300)


def rotate(video_array, mdl: torch.nn.Module = None, device=torch.device("cpu")):
    if mdl is None:
        mdl = load_mdl(device=device)
    mdl.to(device)
    _im = trans(video_array[0])[None, :].float().to(device)
    with torch.no_grad():
        pred = mdl(_im).detach().numpy().argmax()
    print(pred)
    if pred is 0:
        return video_array

    angle = [0, 90, 180, 270][pred]
    result = []
    for array in video_array:
        result.append(imutils.rotate_bound(array, 360 - angle))

    return result
