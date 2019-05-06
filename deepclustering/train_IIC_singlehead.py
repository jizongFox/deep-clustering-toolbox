from deepclustering.train import Trainer_Initialization
from deepclustering.model import Model
from deepclustering.trainer import IIC_trainer
from torch.utils.data import DataLoader
from typing import Tuple

trainer_manager = Trainer_Initialization()
model: Model = trainer_manager.return_Model()
train_loader, val_loader = trainer_manager.return_DataLoders()  # type: Tuple[DataLoader,DataLoader]
Trainer = trainer_manager.return_trainer()
trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader)
trainer.start_training()
