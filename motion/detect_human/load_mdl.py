#%%
from fastai import *
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.callbacks.hooks import num_features_model
import cv2
from fastai.callbacks.hooks import num_features_model
from fastai.vision import create_head

from motion.detect_human import BBoxDataset

from motion.detect_human.archs.detect_humans_res18_arch2 import HumanBBox

#
# based on the notebook: loading_fastai_dict_into_pytorch_model


def load_mdl(f_mdl=None):
    if f_mdl is None:
        f_mdl = "models/detect_humans_res18_arch2.pt"
    assert os.path.exists(f_mdl)
    print("loading {f_mdl}")
    mdl = HumanBBox()

    mdl.load_state_dict(torch.load(f_mdl), strict=False)
    return mdl

