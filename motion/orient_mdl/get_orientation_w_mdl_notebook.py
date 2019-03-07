#%%
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


#%%
import motion


#%%
im = cv2.imread("coco/divera_trend.png")
plt.imshow(im)


#%%
mdl_pose_resnet = motion.pose_resnet.get_pose_model()


#%%
from motion.detect_human.fastai_utils import *


class OrientMdl(nn.Module):
    def __init__(self, pose_resnet, p=0.5):
        super(OrientMdl, self).__init__()
        self.cnn = nn.Sequential(*list(pose_resnet.children())[:2])
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


#%%
mdl = OrientMdl(mdl_pose_resnet)
mdl.eval()
print("")


#%%
import cv2
import pandas as pd
import imutils

from torch.utils.data import Dataset

from motion.detect_human_ssd import PredictionTransform


class RotationDataset(Dataset):
    def __init__(self, csv_file, type="train"):
        super(RotationDataset).__init__()
        self.df = pd.read_csv(csv_file)
        print("Dataset has {} samples.".format((self.df.shape[0])))
        self.trans = PredictionTransform(300)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, angle_i=None):
        data = dict(self.df.iloc[index, :])
        if angle_i is None:
            angle_i = np.random.randint(3)
        angle = [0, 90, 180, 270][angle_i]
        img = cv2.imread(data["filename"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = imutils.rotate_bound(img, angle)
        return {"data": self.trans(img).float(), "target": angle_i}


rot_dl = DataLoader(RotationDataset("coco/val2017_one_human_train.csv"))
rot_val_dl = DataLoader(RotationDataset("coco/val2017_one_human_val.csv"))


#%%

from neural_pipeline.builtin.monitors.tensorboard import TensorboardMonitor
from neural_pipeline import (
    DataProducer,
    AbstractDataset,
    TrainConfig,
    TrainStage,
    ValidationStage,
    Trainer,
    FileStructManager,
)

# from ..detect_human import HumanBBox, BBoxDataset, IoU

fsm = FileStructManager(base_dir="models/detect_rotv_3/", is_continue=True)


train_dataset = DataProducer(
    [RotationDataset("coco/train2017_one_human.csv")], batch_size=64, num_workers=5
)
validation_dataset = DataProducer(
    [RotationDataset("coco/val2017_one_human_train.csv")], batch_size=64, num_workers=2
)

train_config = TrainConfig(
    [TrainStage(train_dataset), ValidationStage(validation_dataset)],
    # torch.nn.L1Loss(),
    nn.CrossEntropyLoss(),
    torch.optim.SGD(mdl.parameters_small(), lr=2e-2),
)

trainer = (
    Trainer(mdl, train_config, fsm, torch.device("cuda:0")).set_epoch_num(5)
    # .enable_lr_decaying(0.97, 1000)
)

trainer.monitor_hub.add_monitor(TensorboardMonitor(fsm, is_continue=True))
# trainer.monitor_hub.update_metrics({'accuracy':})
# .resume(from_best_checkpoint=False)
trainer.train()


#%%
trans = PredictionTransform(300)
print(nn.Softmax()(mdl(trans(im)[None, :].float().to(torch.device("cuda:0")))))
plt.imshow(im)


#%%
ds = RotationDataset("coco/val2017_one_human_train.csv")


#%%


def plot_im(i, ds):
    sample = ds[i]
    im = sample["data"]
    orignal = im.detach().numpy().transpose(1, 2, 0).astype(np.int)
    plt.imshow(orignal)
    pred = (
        nn.Softmax()(mdl(im[None, :].float().to(torch.device("cuda:0"))))
        .cpu()
        .detach()
        .numpy()
    )
    print(sample["target"], np.argmax(pred), pred)
    plt.show()


for i in range(10):
    plot_im(i, ds)


#%%
good = 0
for i in range(len(ds)):
    sample = ds[i]
    im = sample["data"]

    pred = np.argmax(
        nn.Softmax()(mdl(im[None, :].float().to(torch.device("cuda:0"))))
        .cpu()
        .detach()
        .numpy()
    )
    good += int(pred == sample["target"])
print(good / len(ds))


#%%
PATH = "models/OrientMdl.pth"
torch.save(mdl.state_dict(), PATH)


#%%
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

