import cv2
import numpy as np

from . import BBox, BBoxPreds
from . import config


def crop_to_human(arrays, bbox_results: [BBoxPreds], size=None):
    if not size:
        size = config.get_image_size("both")
    vid_array = []
    for i in range(arrays.shape[0]):
        if not bbox_results[i].incl_human:
            Warning(f"No human found in frame {i}.")
        else:
            vid_array.append(
                _crop_to_human(arrays[i], bbox_results[i].get_human("bbox"), size)
            )
    if vid_array:
        return np.concatenate(vid_array, 0)
    raise NotImplementedError("No person found in video")


def _crop_to_human(array: np.array, bbox: BBox, size: (int, int)):
    x, w, y, h = bbox.get_all()
    src_points = np.float32([[x - w, y - h], [x - w, y + h], [x + w, y + h]])
    dst_points = np.float32([[0, 0], [0, size[1]], [size[0], size[1]]])
    trans = cv2.getAffineTransform(src_points, dst_points)
    input = cv2.warpAffine(
        array, trans, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR
    )
    return input[None, :]
