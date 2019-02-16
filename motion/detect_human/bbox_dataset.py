import os
import logging
import os

from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import numpy as np

from motion import config

logger = logging.getLogger(__name__)


class BBoxDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        super(BBoxDataset).__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        data = dict(self.df.iloc[index, :])
        sample = {
            "image": io.imread(data["filename"]),
            "landmarks": np.array(
                [data["y"], data["x"], data["height"], data["width"]]
            ),
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

