import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import lightning as L
from .utils import Scorer


class BanknotesClassifierModule(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 batch_size: int,
                 lr: float,
                 train_dataset: Dataset,
                 valid_dataset: Dataset,
                 scorer: Scorer,
                 ):
        super(BanknotesClassifierModule, self).__init__()

        # creates self.hprams and adds all the provided parameters in init to it
        # can be used to access hprarams anywhere

        # Note : learning rate is always called (lr) in self.hparams

        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.lr = lr
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.model = model
        self.scorer = scorer
        for k,v in self.scorer.train_scorers.items():
            v=v.to(device=self.device)
        for k,v in self.scorer.valid_scorers.items():
            v=v.to(device=self.device)
        self.softmax = nn.Softmax(dim=1)
        self.save_hyperparameters(ignore=['model', 'train_dataset', 'valid_dataset', "scorer"])
        # self._log_hyperparams = False
        

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.AdamW(params=params, lr=self.hparams.lr)
        return optimizer

    def get_accuracy(self, outputs, labels):
        res = torch.argmax(outputs, dim=1)
        return torch.sum(res == labels)

    def shared_step(self, batch, batch_idx, stage):
        images, labels = batch
        outs = self.forward(images)
        loss = self.criterion(outs, labels)
        acc = self.get_accuracy(outs, labels)/len(labels)

        """
        to log a value :
            - to make it appear on the progress bar : set prog_bar = True
            - to make it appear for each step : set on_step = True (default for training_step() and training_step_end())
            - to make it appear for each epoch : set on_epoch = True (default for training_epch_end() and validation and test funcs)
        """
        if (stage == 'train'):
            scores = self.scorer.get_training_scores(outs, labels)
            self.log("training_acc", acc, prog_bar=True,
             on_step=True, on_epoch=True)
            self.log("training_loss", loss, prog_bar=True,
                     on_step=True, on_epoch=True)

        elif (stage == 'valid'):
            self.scorer.update_validation_scores(outs, labels)
            # self.log("valid_acc", acc, prog_bar=True, on_epoch=True)
            self.log("valid_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'valid')

    def on_train_epoch_end(self) -> None:
        scores = self.scorer.compute_epoch_training_scores()
        self.log("training_f1", scores["training_f1"], prog_bar=True,
                    on_step=False, on_epoch=True)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self) -> None:
        scores = self.scorer.compute_epoch_validation_scores()
        self.log("validation_acc", scores["validation_acc"], prog_bar=True, 
                 on_epoch=True, on_step=False)
        self.log("validation_f1", scores["validation_f1"], prog_bar=True,
                 on_step=False, on_epoch=True)
        return super().on_validation_epoch_end()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, shuffle=False, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset, shuffle=False, batch_size=self.batch_size)
