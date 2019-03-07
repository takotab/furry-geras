import numpy as np
import torch
import cv2

from . import box_utils
from .data_preprocessing import PredictionTransform
from .utils_misc import Timer
from ..utils import BBoxPreds, BBox

from torch.utils.data import Dataset, DataLoader


class Predictor:
    def __init__(
        self,
        net,
        size,
        mean=0.0,
        std=1.0,
        nms_method=None,
        iou_threshold=0.45,
        filter_threshold=0.01,
        candidate_size=200,
        sigma=0.5,
        device=None,
        class_names=None,
    ):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.class_names = class_names

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None, video=False):
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        height, width, _ = image.shape

        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        return self.procces_images(boxes, scores, height, width, prob_threshold, top_k)

    def predict_video(self, video, top_k, prob_threshold, bs=16):
        vid_dl = DataLoader(Video(video, self.transform), batch_size=bs)
        scores, boxes = [], []
        with torch.no_grad():
            for images in vid_dl:
                self.timer.start()
                _score, _box = self.net.forward(images)
                print("Inference time: ", self.timer.end())
                scores.append(_score)
                boxes.append(_box)
        scores, boxes = torch.cat(scores, 0), torch.cat(boxes, 0)
        # print(scores.shape)
        height, width, _ = video[0].shape
        bbox_preds = []
        for i in range(scores.shape[0]):
            _bbox_pred = self.procces_images(
                boxes, scores, height, width, prob_threshold, i
            )
            bbox_preds.append(_bbox_pred)
        return bbox_preds

    def procces_images(self, boxes, scores, height, width, prob_threshold, top_k, i=0):
        cpu_device = torch.device("cpu")
        boxes = boxes[i]
        scores = scores[i]
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(
                box_probs,
                self.nms_method,
                score_threshold=prob_threshold,
                iou_threshold=self.iou_threshold,
                sigma=self.sigma,
                top_k=top_k,
                candidate_size=self.candidate_size,
            )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return BBoxPreds(
            picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
        )


class Video(Dataset):
    def __init__(self, numpy_video: [np.array], transform=lambda o: o):
        super(Video).__init__()
        self.b = numpy_video
        self.t = transform

    def __len__(self):
        return len(self.b)

    def __getitem__(self, index):
        return self.t(cv2.cvtColor(self.b[index], cv2.COLOR_BGR2RGB))

