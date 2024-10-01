import torch.nn as nn
import lightning as L
from .utils import callbacks

def get_trainer(config: dict):
    trainer = L.Trainer(**config)
    trainer.callbacks = callbacks
    return trainer