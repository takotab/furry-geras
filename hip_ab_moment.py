#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms

from lib import models
from lib.core import inference

#%%
model = models.get_fully_pretrained_pose_net().cuda()
model.eval()


#%%


def pose_estimate(img_file, resize=256):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    # cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    data_numpy = cv2.imread(
        str(image_file), cv2.IMREAD_COLOR  # | cv2.IMREAD_IGNORE_ORIENTATION
    )
    print(data_numpy.shape)
    data_numpy = cv2.resize(data_numpy, (resize, resize))
    print(data_numpy.shape)
    input = transform(data_numpy).cuda()
    output = model(input[None, :, :, :]).detach().cpu().numpy()
    print(output.shape)
    pred, val = inference.get_max_preds(output)
    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(
        cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB),
        cmap="gray",
        interpolation="bicubic",
    )

    for p in pred[0]:
        plt.plot(
            (p[0] / output.shape[-1]) * resize,
            p[1] / output.shape[-1] * resize,
            marker="o",
            color="r",
        )

    #         break
    plt.axis("off")
    plt.show()
    fig1.savefig(img_file.replace(".jpg", "_marked.png"), bbox_inches="tight")
    return pred


def main():
    hip_ab_force = HipAbForce()

    image_files = ["data/roos2.jpg", "data/tako.jpg", "data/roos.jpg", "data/tako2.jpg"]
    for image_file in image_files:
        pred = pose_estimate(image_file)
        print(hip_ab_force(pred))


#%%
class HipAbForce(object):
    def __init__(self):

        self.keypoints = {
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
        self.sides = ["left", "right"]
        self.other_side = {
            side: other for side, other in zip(self.sides, self.sides[::-1])
        }

    def zp_wo_leg(self, hip, other_hip, shoulder, other_shoulder):
        y = np.sum([hip[1] * 2.0, other_shoulder[1]]) / 3
        x = np.mean([hip[0], other_hip[0]])
        return hip, (x, y)

    def m_f_muscle(self, hip, fz, weight=75):
        return weight * abs(hip[0] - fz[0])

    def hip_ab_force(self, pred):
        result = {key: 0 for key in self.sides}
        points = {key: loc for key, loc in zip(self.keypoints.values(), pred[0])}

        for side in self.sides:
            result[side] = self.m_f_muscle(
                self.zp_wo_leg(self.points_of_intrest(points, side))
            )
        return result

    def points_of_intrest(self, points, side):
        hip = points[side + "_hip"]
        shoulder = points[side + "_shoulder"]
        other_hip = points[self.other_side[side] + "_hip"]
        other_shoulder = points[self.other_side[side] + "_shoulder"]
        return hip, other_hip, shoulder, other_shoulder
