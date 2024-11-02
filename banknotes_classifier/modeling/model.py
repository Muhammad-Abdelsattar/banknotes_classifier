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

class EfficientNetClassifier(BaseClassifier):
    def _prepare_model(self,
                       pretrained_backbone=True):
        efficientnet = models.efficientnet_b0(pretrained=pretrained_backbone)
        efficientnet.classifier[-1] = nn.Linear(in_features=efficientnet.classifier[-1].in_features,out_features=self.num_classes)
        return efficientnet

class ExportReadyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    
    def normalize_image(self,x):
        x = x / 255.0
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return x

    def forward(self,x):
        x = self.normalize_image(x)
        return self.softmax(self.model(x))