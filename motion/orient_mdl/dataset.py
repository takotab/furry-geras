import pandas as pd
import cv2
import numpy as np
import imutils

from motion.utils.import_pytorch import *
from motion.detect_human_ssd import PredictionTransform


class RotationDataset(Dataset):
    def __init__(self, csv_file, type="train"):
        super(RotationDataset).__init__()
        self.df = pd.read_csv(csv_file)
        print("Dataset has {} samples.".format((self.df.shape[0])))
        self.trans = PredictionTransform(300)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, angle_i=None):
        data = dict(self.df.iloc[index, :])
        if angle_i is None:
            angle_i = np.random.randint(3)
        angle = [0, 90, 180, 270][angle_i]
        img = cv2.imread(data["filename"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = imutils.rotate_bound(img, angle)
        return {"data": self.trans(img).float(), "target": angle_i}
