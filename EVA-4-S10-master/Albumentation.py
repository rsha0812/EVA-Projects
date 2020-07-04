
# imports
import numpy as np
import albumentations as alb
import albumentations.pytorch as alb_pytorch
from albumentations import Compose, RandomCrop, Normalize, Cutout,HorizontalFlip, VerticalFlip, Resize, Rotate
from albumentations.pytorch import ToTensor



class album_compose:
    def __init__(self):
        self.alb_transform = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=(-90, 90)),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor()
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.alb_transform(image=img)['image']
        return img







