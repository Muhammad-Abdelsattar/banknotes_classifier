import albumentations as A
from albumentations.pytorch import ToTensorV2


img_transforms = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])