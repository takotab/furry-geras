import os
import logging
import os

from pathlib import Path
import pandas as pd

# from torch.nn import M
from PIL import Image
import numpy as np
import cv2
from ast import literal_eval
import albumentations

from motion import config

logger = logging.getLogger(__name__)
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class Sample(object):
    def __init__(self, datadct):
        self.bbox = None
        for k, v in datadct.items():
            setattr(self, k, v)

    def _make_crop_sizes(self, dct):
        if self.bbox:
            x, y, bbox = self.crop_for_training()
        else:
            pass

    def crop_for_training(self):
        x, y, w, h = self.bbox
        x, x_bbox = self._resize_one_dims(x, w, self.width, "x")
        y, y_bbox = self._resize_one_dims(y, h, self.height, "y")

        bbox = [x_bbox[0], y_bbox[0], x_bbox[1], y_bbox[1]]

        return x, y, bbox

    def _resize_one_dims(self, start, length, orign_len, debug_x="x", pick_start=None):
        start, length, orign_len, size = (
            int(start),
            int(length),
            int(orign_len),
            self._size,
        )
        self.debug_stats[debug_x]["in"] = (start, length, orign_len, size)
        pick_start = self.handle_pick_start(pick_start)

        bbox = None
        if orign_len > size:
            if length > size:
                m = start + length / 2
                o_s = np.min([int(m - (size / 2)), orign_len - size])
                o_e = o_s + size

                n_s = 0
                n_e = size
                bbox = [0, size]
            else:
                [n_s, n_e, o_s, o_e] = smaller_length_bigger_or_len(
                    start, length, orign_len, size, pick_start
                )
        elif orign_len == size:
            n_s, n_e, o_s, o_e = 0, 500, 0, 500
        else:
            o_s = 0
            o_e = orign_len
            n_s = int(pick_start(0, size - orign_len))
            n_e = n_s + orign_len
        if bbox is None:
            bbox = [start - o_s + n_s, start + length - o_s + n_s]
        self.debug_stats[debug_x]["out"] = [n_s, n_e, o_s, o_e], bbox
        return [n_s, n_e, o_s, o_e], bbox

    def handle_pick_start(self, pick_start):
        if pick_start:
            return lambda o, u: pick_start([o, u])
        elif self.type == "train":
            return np.random.randint
        else:
            return lambda o, u: np.mean([o, u])

    def keep_a_bbox(self, item, index):
        if len(item["bboxes"]) == 4:
            item["bboxes"] = [item["bboxes"]]
        elif len(item["bboxes"]) == 0:
            Warning("len(item['bboxes']) == 0 on index {}".format(str(index)))
            item["bboxes"] = [[0, 0, 0, 0]]

        assert (
            len(item["bboxes"]) == 1
        ), "bbox should have size 1 and but item[bbox]: {}\n Error is with index {}, {}, {}".format(
            str(item["bboxes"]), str(index), self.type, self.aug
        )
        return item


def smaller_length_bigger_or_len(start, length, orign_len, size, pick_start):
    if orign_len - size < start:
        e = start
    else:
        e = orign_len - size
        e = np.min([start, orign_len - size])

    s = np.max([0, start + length - size])
    s, e = [np.min([se, orign_len - size]) for se in [s, e]]
    if s != e:
        o_s = int(pick_start(s, e))
    else:
        o_s = e
    o_e = size + o_s
    n_s = 0
    n_e = size
    return [n_s, n_e, o_s, o_e]
