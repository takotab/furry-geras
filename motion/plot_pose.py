from matplotlib import pyplot as plt
import cv2

from . import config


def plot_pose(data_numpy, points=None):
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
        f = "output/temp_frame_" + str(i) + ".png"
        plt.savefig(f)
        files.append(f)
        plt.close()
    return files


def get_scale(key):
    return config.resize[key] / config.output_shape(key)
