import datetime
import os
import numpy as np

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

from .. import config


def make_video(images, outvid=None, fps=30, size=None, is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    if outvid is None:
        outvid = "output/temp_" + datetime.datetime.now().isoformat() + ".mp4"
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if type(image) == str:
            if not os.path.exists(image):
                raise FileNotFoundError(image)
            img = imread(image)
        else:
            img = image
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


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
