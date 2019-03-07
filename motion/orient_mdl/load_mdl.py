import torch
from .arch import OrientMdl

MDL_DIR = "models/OrientMdlv_2.pth"


def load_mdl(mdl_dir: str = None, device=torch.device("cpu")):
    if mdl_dir is None:
        mdl_dir = MDL_DIR
    model = OrientMdl()
    model.load_state_dict(torch.load(mdl_dir))
    model.eval()
    return model.to(device)

