import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import torch

import motion


def main(video_dir):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    start_time = time.time()
    array = motion.get_video_array(video_dir)
    print("geting video", time.time() - start_time)
    pred = motion.get_pose(array, device="cpu")
    print("geting get_pose", time.time() - start_time)
    video = motion.plot_pose(array, pred)
    print("geting plot_pose", time.time() - start_time)


if __name__ == "__main__":
    video_dir = "coco/divera_trend.mp4"
    main(video_dir)
