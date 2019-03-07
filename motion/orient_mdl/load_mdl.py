import torch
from .arch import OrientMdl

mdl_dir = "models/OrientMdl.pth"


def load_mdl(mdl_dir="models/OrientMdl.pth", device=torch.device("cpu")):

    model = OrientMdl()
    model.load_state_dict(torch.load(mdl_dir))
    model.eval()
    return model.to(device)

