from ..utils.import_pytorch import *
from motion.utils.fastai_utils import AdaptiveConcatPool2d, Flatten
from motion import pose_resnet


class OrientMdl(nn.Module):
    def __init__(self, mdl_pose_resnet=None, p=0.5):
        super(OrientMdl, self).__init__()
        if mdl_pose_resnet is None:
            mdl_pose_resnet = pose_resnet.get_pose_model()

        self.cnn = nn.Sequential(*list(mdl_pose_resnet.children())[:2])
        self.pool = nn.Sequential(*[AdaptiveConcatPool2d(), Flatten()])
        self.head = nn.Sequential(
            *[nn.BatchNorm1d(128), nn.Dropout(p), nn.Linear(128, 4)]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        x = self.head(x)
        return x

    def parameters_small(self):
        return [*list(self.pool.parameters()), *list(self.head.parameters())]
