import cv2
import numpy as np
import torch
import time
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from . import pose_resnet

output_shape = (0, 0)


def get_pose(numpy_video, *args, **kwargs):
    pred = pose_estimate(numpy_video, *args, **kwargs)
    return pred


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])


class Video(Dataset):
    def __init__(self, numpy_video: [np.array]):
        super(Video).__init__()
        self.b = numpy_video

    def __len__(self):
        return len(self.b)

    def __getitem__(self, index):
        return transform(self.b[index])


def pose_estimate(data_array, vidpose, data=None, batch_size=16):
    start_time = time.time()

    if data is None:
        data = DataLoader(Video(data_array), batch_size=batch_size)
    pred, val = [], []
    print("batch_size", batch_size, len(data))
    for input in data:
        output = vidpose.pose_mdl(input.to(vidpose.device))
        _pred, _val = pose_resnet.inference.get_max_preds(output.detach().cpu().numpy())
        pred.append(_pred)
        val.append(val)
        print(time.time() - start_time)
    return np.concatenate(pred)


def read_image(img_file):
    return cv2.imread(
        str(img_file), cv2.IMREAD_COLOR  # | cv2.IMREAD_IGNORE_ORIENTATION
    )
