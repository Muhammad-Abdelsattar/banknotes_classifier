import torch 
import torch.nn as nn 
import torchvision.models as models

class BaseClassifier(nn.Module):
    def __init__(self,
                 num_classes: int,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.model = self._prepare_model(pretrained_backbone=pretrained_backbone)

    def _prepare_model(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self,x):
        return self.model(x)


class MobileNetClassifier(BaseClassifier):
    def _prepare_model(self,
                       pretrained_backbone=True):
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained_backbone)
        mobilenet.classifier[-1] = nn.Linear(in_features=1024,out_features=self.num_classes)
        return mobilenet


class Regnet400Classifier(BaseClassifier):
    def _prepare_model(self,
                       pretrained_backbone=True):
        regnet = models.regnet_y_400mf(pretrained=pretrained_backbone)
        regnet.fc = nn.Linear(in_features=regnet.fc.in_features,out_features=self.num_classes)
        return regnet