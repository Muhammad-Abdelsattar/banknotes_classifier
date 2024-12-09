import numpy as np
from sklearn.metrics import confusion_matrix

def collate_model_outputs(model_outs):
    predictions = [pred for pred,_ in model_outs]
    scores = [score for _, score in model_outs]
    return predictions, scores

def compute_accuracy(model_outs, labels):
    predictions, _ = collate_model_outputs(model_outs)
    accuracy = np.mean(np.array(predictions) == np.array(labels)).item()
    return accuracy

def compute_avg_confidence_score(model_outs):
    _, scores = collate_model_outputs(model_outs)
    avg_score = np.mean(scores)
    return avg_score

def get_confusion_matrix(model_outs, labels):
    predictions, _ = collate_model_outputs(model_outs)
    return confusion_matrix(labels, predictions)