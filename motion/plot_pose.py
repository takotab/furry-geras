from matplotlib import pyplot as plt
import cv2
import os
from .utils import filename_maker
from . import config


def plot_pose(data_numpy, points=None, name=None):
    files = []
    for i in range(data_numpy.shape[0]):
        plt.imshow(
            cv2.cvtColor(data_numpy[i], cv2.COLOR_BGR2RGB),
            cmap="gray",
            interpolation="bicubic",
        )
        if points is not None:
            for p in points[i]:
                plt.plot(
                    (p[0] * get_scale("height")),
                    (p[1] * get_scale("width")),
                    marker="o",
                    color="r",
                )
        if name is None:
            name = filename_maker()
        f = os.path.join(name, "temp_frame_" + str(i) + ".png")
        plt.axis("off")
        plt.savefig(f, bbox_inches="tight")
        files.append(f)
        plt.close()
    return files


def get_scale(key):
    return config.get_image_size(key) / config.output_shape(key)
