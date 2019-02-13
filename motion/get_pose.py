import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from . import pose_resnet

output_shape = (0, 0)


def get_pose(numpy_video):
    pred = pose_estimate(numpy_video)
    return pred


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])


class Video(Dataset):
    def __init__(self, numpy_video):
        super(Video).__init__()
        self.b = numpy_video

    def __len__(self):
        return self.b.shape[0]

    def __getitem__(self, index):
        return transform(self.b[index])


def pose_estimate(data_array):

    model = get_model()
    data = DataLoader(Video(data_array), batch_size=32)

    pred, val = [], []
    for input in data:
        output = model(input.cuda()).detach().cpu().numpy()
        _pred, _val = pose_resnet.inference.get_max_preds(output)
        pred.append(_pred)
        val.append(val)
    return np.concatenate(pred)


def get_model(layers=50):
    return pose_resnet.get_fully_pretrained_pose_net().cuda()


def read_image(img_file):
    return cv2.imread(
        str(img_file), cv2.IMREAD_COLOR  # | cv2.IMREAD_IGNORE_ORIENTATION
    )
