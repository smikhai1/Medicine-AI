from albumentations import *
from albumentations.torch import ToTensor
import numpy as np

import random
import numpy as np
# from composition import Compose, OneOf, GrayscaleOrColor
# import functional as F
from imgaug import augmenters as iaa
from scipy.ndimage import label

def pre_transform(resize):
    transforms = []
    transforms.append(Resize(resize, resize))
    return Compose(transforms)

def post_transform():
    return Compose([
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)),
        ToTensor()])

def mix_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        post_transform()
    ])

def test_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        post_transform()]
    )
