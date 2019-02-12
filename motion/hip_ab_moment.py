#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms


keypoints = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}
sides = ["left", "right"]
other_side = {side: other for side, other in zip(sides, sides[::-1])}


def hip_ab_moment(pred):
    result = {key: 0 for key in sides}
    points = {key: loc for key, loc in zip(keypoints.values(), pred[0])}

    for side in sides:
        result[side] = _m_f_muscle(_zp_wo_leg(_points_of_intrest(points, side)))
    return result


def _zp_wo_leg(hip, other_hip, shoulder, other_shoulder):
    y = float(np.sum([hip[1] * 2.0, other_shoulder[1]]) / 3)
    x = float(np.mean([hip[0], other_hip[0]]))
    return hip, (x, y)


def _m_f_muscle(hip, fz, weight=75):
    return weight * abs(hip[0] - fz[0])


def _points_of_intrest(self, points, side):
    hip = points[side + "_hip"]
    shoulder = points[side + "_shoulder"]
    other_hip = points[other_side[side] + "_hip"]
    other_shoulder = points[other_side[side] + "_shoulder"]
    return hip, other_hip, shoulder, other_shoulder
