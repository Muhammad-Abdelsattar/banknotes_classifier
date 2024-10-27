import os
import cv2
import numpy as np
from .inference_pipeline import InferencePipeline

def collate_model_outputs(model_outs):
    predictions = [pred for _, pred in model_outs]
    scores = [score for score, _ in model_outs]
    return predictions, scores

def compute_accuracy(model_outs, labels):
    predictions, _ = collate_model_outputs(model_outs)
    accuracy = np.mean(np.array(predictions) == np.array(labels)).item()
    return accuracy

def compute_avg_score(model_outs):
    _, scores = collate_model_outputs(model_outs)
    avg_score = np.mean(scores)
    return avg_score