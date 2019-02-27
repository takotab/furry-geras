#%%
from fastai import *
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.callbacks.hooks import num_features_model
import cv2
from fastai.callbacks.hooks import num_features_model
from fastai.vision import create_head

from motion.detect_human import BBoxDataset

# based on the notebook: loading_fastai_dict_into_pytorch_model


class FlukeDetector2(nn.Module):
    def __init__(self, arch=models.resnet18):
        super().__init__()
        self.cnn = create_body(arch)
        self.head = create_head(num_features_model(self.cnn) * 2, 4)

    def forward(self, im):
        x = self.cnn(im)
        x = self.head(x)
        return x.sigmoid_()


def load_mdl(f_mdl=None):
    if f_mdl is None:
        f_mdl = "models/fastai_bbox_detect_humans_res18_arch2.pth"
    assert os.path.exists(f_mdl)
    weights = torch.load(f_mdl)["model"]
    mdl = FlukeDetector2(arch=models.resnet50)
    mdl.load_state_dict(state_dict=weights, strict=False)
    return mdl


#%%

