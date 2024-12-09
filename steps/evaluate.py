import os
import glob
import torch
from torch.utils.data import DataLoader
from banknotes_classifier.evaluation.evaluation import compute_accuracy, compute_avg_confidence_score, get_confusion_matrix
from banknotes_classifier.evaluation.inference_pipeline import InferencePipeline
from banknotes_classifier.evaluation.utils import write_metrics, write_confusion_matrix_plot
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
    avg_confidence = compute_avg_confidence_score(model_outs)
    confusion_matrix = get_confusion_matrix(model_outs, labels)
    metrics_dict = {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
    }
    write_metrics(metrics_dict,"reports/evaluation/metrics.json")
    write_confusion_matrix_plot(confusion_matrix,"reports/evaluation/plots/confusion_matrix.png")
    

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
