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

from ..detect_human import HumanBBox, BBoxDataset

SZ = 500


def train(num_epochs=3):
    fsm = FileStructManager(base_dir="models", is_continue=False)
    model = HumanBBox()

    train_dataset = DataProducer(
        [
            BBoxDataset(
                "coco/train2017_one_human.csv", size=SZ, type="train", fastai_out=False
            )
        ],
        batch_size=8,
        num_workers=2,
    )
    validation_dataset = DataProducer(
        [
            BBoxDataset(
                "coco/val2017_one_human_val.csv", size=SZ, type="val", fastai_out=False
            )
        ],
        batch_size=4,
        num_workers=2,
    )

    train_config = TrainConfig(
        [TrainStage(train_dataset), ValidationStage(validation_dataset)],
        torch.nn.L1Loss(),
        torch.optim.SGD(model.head.parameters(), lr=5e-2, momentum=0.8),
    )

    trainer = Trainer(model, train_config, fsm, torch.device("cuda:0")).set_epoch_num(
        num_epochs
    )
    trainer.monitor_hub.add_monitor(TensorboardMonitor(fsm, is_continue=False))
    # .add_monitor(LogMonitor(fsm)

    trainer.train()
