import os
import glob
import torch
from torch.utils.data import DataLoader
from banknotes_classifier.evaluation.evaluation import compute_accuracy, compute_avg_score
from banknotes_classifier.evaluation.inference_pipeline import InferencePipeline
from banknotes_classifier.evaluation.utils import write_metric
from banknotes_classifier.data.dataset import TestBanknotesDataset

def evaluate(config: dict):
    pipeline = InferencePipeline(model_path=config["export"]["model_path"])
    test_dataset_path = config["evaluation"]["test_dataset_path"]
    test_images = glob.glob(os.path.join(test_dataset_path, "*", "*.jpg"))
    test_dataset = TestBanknotesDataset(test_images)
    model_outs = []
    labels = []
    for image, label in test_dataset:
        model_outs.append(pipeline(image))
        labels.append(label)
    accuracy = compute_accuracy(model_outs, labels)
    write_metric("accuracy", accuracy,"reports/metrics/evalluation.json")
    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException
    from typing import Any
    
    try:
        config: dict[str, Any] = OmegaConf.load("params.yaml")
        seed = config.get("random_seed", 42)
        torch.manual_seed(seed)
        evaluate(config)
    except FileNotFoundError:
        print("Error: params.yaml file not found.")
    except OmegaConfBaseException as e:
        print(f"Error loading configuration: {e}")
