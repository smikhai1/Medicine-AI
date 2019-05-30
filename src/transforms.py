import torch
from torchvision.transforms import Compose
import random
import numpy as np
# from composition import Compose, OneOf, GrayscaleOrColor
# import functional as F
# from imgaug import augmenters as iaa
from scipy.ndimage import label

class ToTensor(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if len(image.shape) != 3:
            image = image[:, :, None] # add the channel dimension

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).to(torch.float32),
                'mask': torch.from_numpy(mask)
                }


class ToTensor3D(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image[None, :, :, :]

        return {'image': torch.from_numpy(image).to(torch.float32),
                'mask': torch.from_numpy(mask)
                }

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h, left: left + new_w]

        return {'image': image, 'mask': mask}

class CenterCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        centr_h, centr_w = h // 2, w // 2

        start_h = centr_h - new_h//2
        end_h = centr_h + new_h//2

        start_w = centr_w - new_w // 2
        end_w = centr_w + new_w // 2


        image = image[start_h:end_h,
                      start_w:end_w
                     ]

        mask = mask[start_h:end_h,
                    start_w:end_w
                   ]

        assert len(image.shape)==3 or len(mask.shape)==3, 'One dimension was removed'
        return {'image': image, 'mask': mask}

def pre_transform(resize):
    transforms = []
    transforms.append(CenterCrop(resize))
    return Compose(transforms)

def post_transform():
    return Compose([
        #Normalize(
            #mean=(0.485, 0.456, 0.406),
            #std=(0.229, 0.224, 0.225)),
        ToTensor3D()])

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
