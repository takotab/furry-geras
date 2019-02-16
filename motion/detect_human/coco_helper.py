# ------------------------------------------------------------------------------
# Written by Tako Tabak (tako.tabak@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image


# from .JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class CocoHelper:
    def __init__(self, coco_ann_dir):

        self.pixel_std = 200
        self.data_folder = "val2017"
        self.coco = COCO(coco_ann_dir)

        # deal with class names
        cats = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ["__background__"] + cats
        logger.info("=> classes: {}".format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

    def get_one_pic_w_bb(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)
        path = Path(self.image_path_from_index(index))
        width, height = Image.open(path).size

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj["bbox"]
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj["clean_bbox"] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs
        labels, classes = [], {}
        bboxs = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj["category_id"]]
            if cls != 1:
                continue
            labels.append(cls)
            x, y, w, h = obj["clean_bbox"][:4]
            bboxs = [y / height, x / width, (y + h) / height, (x + w) / width]

        if len(labels) != 1:
            return None, None

        return path, bboxs

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = "%012d.jpg" % index

        image_path = os.path.join("coco", self.data_folder, file_name)

        return image_path

