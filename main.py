import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import torch

import motion
from motion import config


def get_video_array(video_dir):

    cap = cv2.VideoCapture(video_dir)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty(
        (frameCount, config.resize["height"], config.resize["width"], 3),
        np.dtype("uint8"),
    )

    fc = 0
    ret = True

    while fc < frameCount and ret:
        ret, b = cap.read()
        buf[fc] = data_resize(b)
        fc += 1

    cap.release()
    return buf


def data_resize(data_array):
    return cv2.resize(data_array, (config.resize["width"], config.resize["height"]))


def main(video_dir):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    start_time = time.time()
    array = get_video_array(video_dir)
    print("geting video", time.time() - start_time)
    pred = motion.get_pose(array, device="cpu")
    print("geting get_pose", time.time() - start_time)
    video = motion.plot_pose(array, pred)
    print("geting plot_pose", time.time() - start_time)


if __name__ == "__main__":
    video_dir = "coco/divera_trend.mp4"
    main(video_dir)
