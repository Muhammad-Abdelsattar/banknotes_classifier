import torch.nn as nn
import lightning as L
from .plugins import build_callbacks, build_logger

def build_trainer(config: dict):
    trainer = L.Trainer(**config["trainer"])
    trainer.callbacks = build_callbacks(config["callbacks"])
    trainer.logger = build_logger(config["logger"])
    return trainer