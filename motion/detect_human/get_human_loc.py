import torch
import numpy as np


def get_human_loc(mdl, video_array):
    vid = video_array.transpose((0, 3, 1, 2))
    vid = torch.from_numpy(vid).float().div(255)

    with torch.no_grad():
        mdl.eval()
        return mdl(vid).detach().numpy()
