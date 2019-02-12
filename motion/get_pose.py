import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms

from . import pose_resnet


def get_model(layers=50):
    return pose_resnet.get_fully_pretrained_pose_net().cuda()


def get_pose():
    pass


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

