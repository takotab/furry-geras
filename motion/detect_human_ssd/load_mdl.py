import torch

from .vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from .mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)

image_path = "tako_bike.jpg"


def load_mdl(model_path=None, label_path=None, device=torch.device("cpu")):
    if model_path is None:
        model_path = "models/mb2-ssd-lite-mp-0_686.pth"
    if label_path is None:
        label_path = "models/voc-model-labels.txt"

    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, device=device)
    net.load(model_path)

    predictor = create_mobilenetv2_ssd_lite_predictor(
        net, candidate_size=200, device=device, class_names=class_names
    )
    return predictor
