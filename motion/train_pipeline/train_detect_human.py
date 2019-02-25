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


fsm = FileStructManager(base_dir="coco", is_continue=False)
model = HumanBBox()

train_dataset = DataProducer([BBoxDataset()], batch_size=8, num_workers=2)
validation_dataset = DataProducer([BBoxDataset()], batch_size=4, num_workers=2)

train_config = TrainConfig(
    [TrainStage(train_dataset), ValidationStage(validation_dataset)],
    torch.nn.L1Loss(),
    torch.optim.SGD(model.head.parameters(), lr=1e-2, momentum=0.8),
)

trainer = Trainer(model, train_config, fsm, torch.device("cuda:0")).set_epoch_num(50)
trainer.monitor_hub.add_monitor(TensorboardMonitor(fsm, is_continue=False)).add_monitor(
    LogMonitor(fsm)
)
trainer.train()
