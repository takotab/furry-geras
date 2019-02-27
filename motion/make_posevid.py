import time

import torch

from . import pose_resnet
from . import get_video_array, get_pose, plot_pose, make_video, filename_maker, config


class Video2PoseVid(object):
    def __init__(self, device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.pose_mdl = get_model()
        self.pose_mdl.to(device)

    def make_posevid(self, video_dir):
        start_time = time.time()
        name = filename_maker(video_dir)
        array = get_video_array(video_dir, config.get_image_size("both"))

        print("geting video", time.time() - start_time)
        pred = get_pose(array, self)
        print("geting get_pose", time.time() - start_time)
        pose_imgs_dir = plot_pose(array, pred, name=name)
        print("geting plot_pose", time.time() - start_time)
        posevid_dir = make_video(pose_imgs_dir, name=name)
        print("saved at ", posevid_dir)
        return posevid_dir


def make_posevid(*args, **kwargs):
    vp = Video2PoseVid()
    vp.make_posevid(*args, **kwargs)


def get_model(layers=50):
    return pose_resnet.get_fully_pretrained_pose_net()
