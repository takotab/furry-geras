import time

import torch

import motion
from motion import detect_human_ssd
from . import pose_resnet
from . import get_video_array, plot_pose, make_video, filename_maker, config
from .utils import Timer
from . import orient_mdl


class Video2Pose(object):
    def __init__(self, device=None):
        self.timer = Timer()
        self.timer.start("init")
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.ssd_mdl = detect_human_ssd.load_mdl(device=self.device)

        self.orient_mdl = orient_mdl.load_mdl(device=self.device)

        self.pose_mdl = pose_resnet.load_mdl(device=self.device)

        self.timer.end(key="init", msg="initializing")

    # def get_pose

    def make_posevid(self, video_dir, plot_pose=True):
        self.timer.start()
        self.name = filename_maker(video_dir)
        array = get_video_array(video_dir)  # , config.get_image_size("both"))
        self.timer.lap(msg="video")

        array = orient_mdl.rotate(array, self.orient_mdl)
        self.timer.lap(msg="rotating video")

        bbox_results = detect_human_ssd.predict_video(array, self.ssd_mdl)
        self.timer.lap(msg="human_loc")

        human_array = motion.crop_to_human(array, bbox_results)
        self.timer.lap(msg="crop human")

        pred = pose_resnet.get_pose(array, self.pose_mdl)
        self.timer.lap(msg="get_pose")

        if plot_pose:
            return self._make_plot_pose(human_array, pred)
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

