
# imports
import numpy as np
from albumentations import Compose, RandomCrop, Normalize, Cutout,HorizontalFlip, PadIfNeeded, HueSaturationValue,ShiftScaleRotate
from albumentations.pytorch import ToTensor
import cv2


class album_compose:
    def __init__(self):
        self.alb_transform = Compose([
            PadIfNeeded(min_height=32, min_width=32, border_mode=cv2.BORDER_CONSTANT, value=4, always_apply = False, p=1.0),
            HueSaturationValue(hue_shift_limit=4, sat_shift_limit=13, val_shift_limit=9),
            RandomCrop(32,32),
            HorizontalFlip(p=0.5),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=13, p=0.6),
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







