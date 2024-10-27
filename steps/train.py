import os
import glob
import torch
from torch.utils.data import random_split
from banknotes_classifier.training.trainer import build_trainer
from banknotes_classifier.modeling.model import *
from banknotes_classifier.modeling.lit_module import BanknotesClassifierModule
from banknotes_classifier.modeling.utils import Scorer
from banknotes_classifier.data.dataset import BanknotesDataset
from banknotes_classifier.data.augmentations import img_transforms


def train(config: dict):
    model = MobileNetClassifier(num_classes=14,pretrained_backbone=True)
    # model = Regnet400Classifier(num_classes=14,pretrained_backbone=True)
    scorer = Scorer(num_classes=14)
    images = glob.glob(os.path.join(config["data"]["dataset_path"], "*", "*.jpg"))
    train_images, valid_images = random_split(images, [0.8, 0.2])
    train_dataset = BanknotesDataset(images_paths=train_images, img_transforms=img_transforms)
    valid_dataset = BanknotesDataset(images_paths=valid_images, img_transforms=img_transforms)
    module = BanknotesClassifierModule(model=model,
                                       batch_size=config["training"]["batch_size"],
                                       lr=config["training"]["lr"],
                                       train_dataset=train_dataset,
                                       valid_dataset=valid_dataset,
                                       scorer=scorer)
    trainer = build_trainer(config["training"])
    trainer.fit(module)
    return


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException
    from typing import Any
    
    try:
        config: dict[str, Any] = OmegaConf.load("params.yaml")
        seed = config.get("random_seed", 42)
        torch.manual_seed(seed)
        train(config)
    except FileNotFoundError:
        print("Error: params.yaml file not found.")
    except OmegaConfBaseException as e:
        print(f"Error loading configuration: {e}")


