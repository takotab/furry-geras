import cv2
import numpy as np
import torch
import time
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from . import inference
from ..utils import Timer

output_shape = (0, 0)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])


def get_pose(numpy_video: [np.array], *args, **kwargs):
    pred = pose_estimate(numpy_video, *args, **kwargs)
    return pred


def pose_estimate(data_array, pose_mdl, batch_size=16):
    timer = Timer().start()
    video_dl = DataLoader(Video(data_array), batch_size=batch_size)

    pred, val = [], []
    for i, input in enumerate(video_dl):
        output = pose_mdl(input)
        _pred, _val = inference.get_max_preds(output.detach().cpu().numpy())
        pred.append(_pred)
        val.append(val)
        timer.lap(msg=f"batch {str(i+1)}/{str(len(video_dl))}")

    return np.concatenate(pred)


class Video(Dataset):
    def __init__(self, numpy_video: [np.array]):
        super(Video).__init__()
        self.b = numpy_video

    def __len__(self):
        return len(self.b)

    def __getitem__(self, index):
        return transform(self.b[index])


def read_image(img_file):
    return cv2.imread(
        str(img_file), cv2.IMREAD_COLOR  # | cv2.IMREAD_IGNORE_ORIENTATION
    )
