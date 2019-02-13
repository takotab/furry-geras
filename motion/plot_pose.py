from matplotlib import pyplot as plt
import cv2

from . import config


def plot_pose(data_numpy, points=None):
    fig1 = plt.gcf()
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

        plt.axis("off")
        plt.draw()
        if i == 0:
            plt.show()
    return fig1


def get_scale(key):
    return config.resize[key] / config.output_shape(key)
