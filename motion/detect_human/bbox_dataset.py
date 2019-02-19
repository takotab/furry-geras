import os
import logging
import os

from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from motion import config

logger = logging.getLogger(__name__)


class BBoxDataset(Dataset):
    def __init__(self, csv_file, size=500, transform=None, type="train"):
        super(BBoxDataset).__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self._size = size
        self.type = type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data = dict(self.df.iloc[index, :])
        img = Image.open(data["filename"])
        crop, bbox = self._make_crop_sizes(data)
        img = img.crop(crop)
        sample = {"image": img, "bboxes": [bbox], "category_id": [0]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _make_crop_sizes(self, dct, train=True):
        x, y, w, h = list(dct["bbox"])
        x_crop, x_bbox = self._select_size_one_dim(x, w)
        y_crop, y_bbox = self._select_size_one_dim(y, h)
        bbox = [x_bbox[0], y_bbox[0], x_bbox[1], y_bbox[1]]
        crop = [x_crop[0], y_crop[0], x_crop[1], y_crop[1]]
        return crop, bbox

    def _select_size_one_dim(self, start, length):
        if self.type == "train":
            center = np.random.randint(start, start + length + 1)
        else:
            center = np.mean([start, start + length + 1])
        crop = [int(center - self._size / 2), int(center + self._size / 2)]
        if start + length > crop[1]:
            length = crop[1] - start
        bbox = [start - crop[0], start - crop[0] + length]
        return crop, bbox

