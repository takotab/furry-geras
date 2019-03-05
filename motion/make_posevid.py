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

    # def get_pose

    def make_posevid(self, video_dir, make_plot_pose=True):
        timer = Timer().start()
        name = filename_maker(video_dir)
        array = get_video_array(video_dir, config.get_image_size("both"))

        timer.lap(msg="geting video")
        pred = get_pose(array, self)
        timer.lap(msg="get_pose")
        if make_plot_pose:
            pose_imgs_dir = plot_pose(array, pred, name=name)
            timer.lap(msg="plot_pose")
            posevid_dir = make_video(pose_imgs_dir, name=name)
            timer.end(msg="making video")
            print("saved at ", posevid_dir)
            return posevid_dir


def make_posevid(*args, **kwargs):
    vp = Video2Pose()
    vp.make_posevid(*args, **kwargs)

