import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import torch
import os
import motion


def main(video_dir):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    motion.make_posevid(video_dir)


if __name__ == "__main__":
    video_dir = "coco/divera_trend.mp4"
    main(video_dir)
    # motion.train_pipeline.train()
    # motion.detect_human.bbox_dataset.get_one_sample_csv()
    os.system("python -m pytest --cov=motion test/")
