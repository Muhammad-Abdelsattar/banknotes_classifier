import torch
import torch.nn as nn
import torchmetrics


class Scorer:
    def __init__(self, num_classes):
        self.train_scorers = {}
        self.valid_scorers = {}
        self.train_scorers["training_acc"] = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.valid_scorers["valdation_acc"] = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.train_scorers["training_f1"] = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes)
        self.valid_scorers["valdation_f1"] = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes)

    def get_training_scores(self, logits, labels):
        scores = {}
        for scorer_name, scorer in self.train_scorers.items():
            scores[scorer_name] = scorer(logits, labels)
        return scores

    def update_validation_scores(self, logits, labels):
        for scorer_name, scorer in self.valid_scorers.items():
            scorer.update(logits, labels)
        return 

    def compute_epoch_training_scores(self):
        scores = {}
        for scorer_name, scorer in self.train_scorers.items():
            scores[scorer_name] = scorer.compute()
        self.reset_training_scores()
        return scores

    def compute_epoch_validation_scores(self):
        scores = {}
        for scorer_name, scorer in self.valid_scorers.items():
            scores[scorer_name] = scorer.compute()
        self.reset_validation_scores()
        return scores

    def reset_training_scores(self):
        for scorer_name,scorer in self.train_scorers.items():
            scorer.reset()
        return
    
    def reset_validation_scores(self):
        for scorer_name,scorer in self.valid_scorers.items():
            scorer.reset()
        return