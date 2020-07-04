
import PIL
import numpy as np
from torchvision import transforms
from utils import visualize_cam
from gradCam import GradCAM
from model import *


def show_map(img,model):
    target_model = model
    gradcam = GradCAM.from_config(model_type='resnet', arch=target_model, layer_name='layer4')
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)[None]
    mask, logit = gradcam(img)
    heatmap, cam_result = visualize_cam(mask, img)
    return heatmap, cam_result


