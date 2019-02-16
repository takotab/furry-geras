import os
import logging
import os

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset


from motion.detect_human import CocoHelper

from motion import config

logger = logging.getLogger(__name__)


def main():
    dataDir = "/home/tako/devtools/furry-geras/coco"
    dataType = "val2017"
    annFile = "{}/instances_{}.json".format(dataDir, dataType)
    coco = CocoHelper(annFile)

    data_dct = {}
    for id in coco.coco.getImgIds():
        path, bbox = coco.get_one_pic_w_bb(id)
        if path is None:
            continue
        data_dct[path.name] = {
            "filename": path,
            "y": bbox[0],
            "x": bbox[1],
            "height": bbox[2],
            "width": bbox[3],
        }

    dataset = pd.DataFrame.from_dict(data_dct).T
    dataset.to_csv("coco/one_human.csv", index=False)


if __name__ == "__main__":
    main()
