import os
import torch
from torch.utils.data import Dataset
import cv2
from .augmentations import img_transforms


class BanknotesDataset(Dataset):
    def __init__(self, 
                 images_paths,
                 img_transforms=img_transforms,
                 ):

        self.images_paths = images_paths
        self.img_transforms = img_transforms

    def get_label_from_path(self, path):
        return int(os.path.split(os.path.split(path)[0])[1])

    def transform_image(self, img):
        if self.img_transforms:
            return self.img_transforms(image=img)['image']
        else:
            return img

    def sample(self, idx):
        img = cv2.imread(self.images_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.get_label_from_path(self.images_paths[idx])
        return img, torch.tensor(label, dtype=torch.int64)

    def __getitem__(self, idx):
        img, label = self.sample(idx)
        img = self.transform_image(img)
        return img, label

    def __len__(self):
        return len(self.images_paths)


class TestBanknotesDataset(Dataset):
    def __init__(self, 
                 images_paths,
                 ):

        self.images_paths = images_paths

    def get_label_from_path(self, path):
        return int(os.path.split(os.path.split(path)[0])[1])

    def sample(self, idx):
        img = cv2.imread(self.images_paths[idx])
        label = self.get_label_from_path(self.images_paths[idx])
        return img, label

    def __getitem__(self, idx):
        img, label = self.sample(idx)
        return img, label

    def __len__(self):
        return len(self.images_paths)