import os
import torch
from banknotes_classifier.modeling.lit_module import BanknotesClassifierModule
from banknotes_classifier.modeling.model import ExportReadyModel, MobileNetClassifier, Regnet400Classifier
from banknotes_classifier.modeling.utils import export_model

def export(config: dict):
    ckpt_path = os.path.join(config["training"]["callbacks"]["ModelCheckpoint"]["dirpath"], 
                             config["training"]["callbacks"]["ModelCheckpoint"]["filename"]+".ckpt")

    exported_model_path = config["export"]["model_path"]
    ckpt = BanknotesClassifierModule.load_from_checkpoint(ckpt_path,
                                                          model=Regnet400Classifier(14,False),
                                                          train_dataset=None,
                                                          valid_dataset=None,
                                                          scorer=None)
    model = ExportReadyModel(model=ckpt.model)
    export_model(model, exported_model_path)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException
    from typing import Any
    
    try:
        config: dict[str, Any] = OmegaConf.load("params.yaml")
        seed = config.get("random_seed", 42)
        torch.manual_seed(seed)
        export(config)
    except FileNotFoundError:
        print("Error: params.yaml file not found.")
    except OmegaConfBaseException as e:
        print(f"Error loading configuration: {e}")
