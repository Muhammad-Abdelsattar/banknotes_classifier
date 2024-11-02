import torch
import torch.nn as nn
import torchmetrics


class Scorer:
    def __init__(self, num_classes):
        self.train_scorers = {}
        self.valid_scorers = {}
        self.train_scorers["training_acc"] = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.valid_scorers["validation_acc"] = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.train_scorers["training_f1"] = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes)
        self.valid_scorers["validation_f1"] = torchmetrics.F1Score(
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


def export_model(model, model_path, input_shape=(3, 240, 320), dynamic_height_and_width=True):
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, *input_shape)
    if (dynamic_height_and_width):
        dynamic_axes = {"input": {0: "batch_size",2: "dim_1", 3: "dim_2"}}
    else:
        dynamic_axes = {"input": {0: "batch_size"}}
    torch.onnx.export(model,
                      dummy_input,
                      model_path,
                      dynamic_axes=dynamic_axes,
                      opset_version=11)