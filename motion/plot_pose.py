from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from .utils import filename_maker
from . import config


def plot_pose(array_lst: [np.ndarray], point_lst=None, name=None, save: bool = True):
    if type(array_lst) == np.ndarray:
        array_lst = [array_lst]
        point_lst = [point_lst]
    size = array_lst[0].shape
    files = []
    for i in range(len(array_lst)):
        plt.imshow(
            cv2.cvtColor(array_lst[i], cv2.COLOR_BGR2RGB),
            cmap="gray",
            interpolation="bicubic",
        )

        if point_lst is not None:
            for p in point_lst[i]:
                plt.plot(
                    (p[0] * get_scale("width")),
                    (p[1] * get_scale("height")),
                    marker="o",
                    color="r",
                )
        if save:
            if name is None:
                name = filename_maker()
            f = os.path.join(name, "temp_frame_" + str(i) + ".png")
            plt.axis("off")
            plt.savefig(f, bbox_inches="tight")
            files.append(f)
            plt.close()
        else:
            plt.show()
    return files


def get_scale(key):
    return config.get_image_size(key) / config.output_shape(key)
