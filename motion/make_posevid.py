import time

import torch

from . import pose_resnet
from . import get_video_array, get_pose, plot_pose, make_video, filename_maker, config
from .utils import Timer


class Video2Pose(object):
    def __init__(self, device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.pose_mdl = pose_resnet.get_pose_model()
        self.pose_mdl.to(device)
        self.timer = Timer()

    # def get_pose

    def make_posevid(self, video_dir, plot_pose=True):
        self.timer.start()
        self.name = filename_maker(video_dir)
        array = get_video_array(video_dir, config.get_image_size("both"))

        self.timer.lap(msg="geting video")
        pred = get_pose(array, self)
        self.timer.lap(msg="get_pose")
        if plot_pose:
            return self._make_plot_pose(array, pred)
        return array, pred

    def _make_plot_pose(self, array, pred):
        self.timer.start()
        pose_imgs_dir = plot_pose(array, pred, name=self.name)
        self.timer.lap(msg="plot_pose")
        posevid_dir = make_video(pose_imgs_dir, name=self.name)
        self.timer.lap(msg="making video")
        print("saved at ", posevid_dir)
        return posevid_dir


def make_posevid(*args, **kwargs):
    vp = Video2Pose()
    vp.make_posevid(*args, **kwargs)

