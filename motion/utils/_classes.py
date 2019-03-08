import torch
import numpy as np
from collections import namedtuple


label_path = "models/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]

locations = ["x_center", "x_width", "y_center", "y_height"]


class BBox(object):
    def __init__(self, bbox):
        self.dct = {key: bbox[i] for i, key in enumerate(locations)}

    def __getitem__(self, key):
        if type(key) is str:
            return self.dct[key]
        elif type(key) is int:
            return self.dct[locations[key]]
        else:
            raise KeyError()

    def get_all(self, conv_type=None):
        if conv_type is None:
            return [self.dct[key] for key in locations]
        return [conv_type(self.dct[key]) for key in locations]

    def __str__(self):
        return str([o for o in self.dct.items()])


class BBoxPreds(object):
    def __init__(self, bboxs: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor):
        self.preds = []
        for i in range(bboxs.shape[0]):
            self.preds.append(
                {
                    "bbox": BBox(bboxs[i, :].detach().numpy()),
                    "label": class_names[int(labels[i].detach().numpy())],
                    "prob": float(probs[i].detach().numpy()),
                }
            )
        self._incl_human()

    def get(self, num: int, key: str):
        return self.preds[num][key]

    def set(self, num: int, key: str, item):
        if type(item) == torch.Tensor:
            item = item.detach().numpy()
        if key == "label" and not type(item) == str:
            item = class_names[item]
        assert type(self.preds[num][key]) == type(item)
        self.preds[num][key] = item

    def __len__(self):
        return len(self.preds)

    def _incl_human(self):
        for i, bbox_pred in enumerate(self):
            if bbox_pred["label"] == "person":
                self.incl_human = True
                self.human_idx = i
                return
        self.incl_human = False

    def get_human(self, key):
        if self.incl_human:
            return self.get(self.human_idx, key)
        print(str(self))
        raise KeyError()

    def __iter__(self):
        for i in range(len(self)):
            yield self.preds[i]

    def __str__(self):
        return str([dct for dct in self.preds]) + str(self.incl_human)


class Model(object):
    def __init__(self, mdl, transform=None, device: torch.device = None, **kwargs):
        if device is None:
            device = torch.device("cpu")
        self.mdl = mdl
        self.device = device
        self.mdl.to(self.device)
        self.transform = transform

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, x):
        if self.transform:
            x = self.transform
        return self.mdl(x.to(self.device))
