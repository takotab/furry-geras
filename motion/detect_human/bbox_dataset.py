import os
import logging
import os

from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from ast import literal_eval

from motion import config

logger = logging.getLogger(__name__)


class BBoxDataset(Dataset):
    def __init__(self, csv_file, size=500, transform=None, type="train"):
        super(BBoxDataset).__init__()
        self.df = pd.read_csv(csv_file, converters={"bbox": literal_eval})
        self.transform = transform
        self._size = int(size)
        self.type = type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data = dict(self.df.iloc[index, :])
        x, y, bbox = self._make_crop_sizes(data)

        img = cv2.imread(data["filename"])
        n_img = np.random.randint(0, 255, size=(self._size, self._size, 3)).astype(
            np.uint8
        )
        n_img[y[0] : y[1], x[0] : x[1], :] = img[y[2] : y[3], x[2] : x[3], :]
        sample = {"image": np.array(n_img), "bboxes": bbox, "category_id": [0]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _make_crop_sizes(self, dct):
        x, y, w, h = dct["bbox"]
        x, x_bbox = self._resize_one_dims(x, w, dct["width"])
        y, y_bbox = self._resize_one_dims(y, h, dct["height"])

        bbox = [x_bbox[0], y_bbox[0], x_bbox[1] - x_bbox[0], y_bbox[1] - y_bbox[0]]

        return x, y, bbox

    def _resize_one_dims(self, start, length, orign_len):
        start, length, orign_len, size = (
            int(start),
            int(length),
            int(orign_len),
            self._size,
        )
        if self.type == "train":
            pick_start = np.random.randint
        else:
            pick_start = lambda o, u: np.mean([o, u])
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
                s = np.min([start, orign_len - size])
                o_s = int(pick_start(0, s))
                o_e = size + o_s
                n_s = 0
                n_e = size
        else:
            o_s = 0
            o_e = orign_len
            n_s = int(pick_start(0, size - orign_len))
            n_e = n_s + orign_len
        if bbox is None:
            bbox = [start - o_s + n_s, start + length - o_s + n_s]
        return [n_s, n_e, o_s, o_e], bbox
