import os
import logging

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from pycocotools.coco import COCO
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def main():

    dataDir = "/home/tako/devtools/furry-geras/coco"
    dataType = "val2017"
    annFile = "{}/instances_{}.json".format(dataDir, dataType)
    coco = COCO(annFile)

    data_dct = {}
    num_image = []
    for i, f in enumerate(Path(dataDir + "/" + dataType).glob("*.jpg")):
        id = int(f.name.replace(".jpg", ""))
        im_ann = coco.loadImgs(id)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        annIds = coco.getAnnIds(imgIds=id, iscrowd=False)
        objs = coco.loadAnns(annIds)

        valid_objs = []
        for obj in objs:
            x, y, w, h = obj["bbox"]
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)

        objs = valid_objs
        bboxs = []
        for obj in objs:
            cls = obj["category_id"]
            if cls != 1:
                continue
            if obj["area"] < 2500:
                continue
            bboxs.append(obj["clean_bbox"])

        if len(bboxs) == 1:
            bbox = bboxs[0]
            data_dct[f.name] = {
                "filename": f,
                "bbox": bbox,
                "height": height,
                "width": width,
            }

    dataset = pd.DataFrame.from_dict(data_dct).T
    dataset.to_csv(f"coco/{dataType}_one_human.csv", index=False)

    print(
        "{} of one human pictures found in {}".format(str(dataset.shape[0]), dataType)
    )


if __name__ == "__main__":
    main()
