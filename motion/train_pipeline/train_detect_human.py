import torch

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

from ..detect_human import HumanBBox, BBoxDataset, IoU

SZ = 500


def train(num_epochs=5):
    fsm = FileStructManager(base_dir="models/UoI/", is_continue=False)
    model = HumanBBox()

    train_dataset = DataProducer(
        [
            BBoxDataset(
                "coco/train2017_one_human.csv", size=SZ, type="train", fastai_out=False
            )
        ],
        batch_size=8,
        num_workers=5,
    )
    validation_dataset = DataProducer(
        [
            BBoxDataset(
                "coco/val2017_one_human_train.csv",
                size=SZ,
                type="val",
                fastai_out=False,
            )
        ],
        batch_size=4,
        num_workers=2,
    )

    train_config = TrainConfig(
        [TrainStage(train_dataset), ValidationStage(validation_dataset)],
        # torch.nn.L1Loss(),
        IoU,
        torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.8),
    )

    trainer = (
        Trainer(model, train_config, fsm, torch.device("cuda:0")).set_epoch_num(
            num_epochs
        )
        # .enable_lr_decaying(0.97, 1000)
    )
    trainer.monitor_hub.add_monitor(TensorboardMonitor(fsm, is_continue=True))
    # .add_monitor(LogMonitor(fsm)
    # .resume(from_best_checkpoint=False)
    trainer.train()
