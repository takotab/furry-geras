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

import fastai
from fastai.vision import *
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .. import config

# from .JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class COCOHumanBBoxDataset(Dataset):
    def __init__(self, coco_ann_dir):
        super(COCOHumanBBoxDataset).__init__()
        self.image_width = config.get_image_size("width")
        self.image_height = config.get_image_size("height")
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
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
        self.make_df()

    def make_df(self):
        
        for id in coco.coco.getImgIds():   
            _humans = 0
            annIds = coco.coco.getAnnIds(imgIds=id, iscrowd=None)
            anns = coco.coco.loadAnns(annIds)
            for obj in anns:
                if obj['category_id'] == 1:
                    _humans +=1
            num_human.append(_humans)
    
    def get_one_pic_w_bb(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

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
        print(objs)
        labels, classes = [], {}
        bboxs = []
        human_found = False
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj["category_id"]]
            if cls != 1:
                continue
            classes[cls] = self.coco.loadCats(cls)[0]["name"]
            x, y, w, h = obj["clean_bbox"][:4]
            labels.append(cls)
            bboxs.append([y, x, y + h, x + w])

        img = open_image(self.image_path_from_index(index))
        bbox = fastai.vision.ImageBBox.create(
            *img.size, bboxs, labels=labels, classes=classes
        )
        return img, bbox

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32
        )
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = "%012d.jpg" % index

        image_path = os.path.join("coco", self.data_folder, file_name)

        return image_path

