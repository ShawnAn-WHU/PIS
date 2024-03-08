import numpy as np
from PIL import ImageFilter, ImageOps

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class GBlur(object):
    '''Gaussian blur.

    Args:
        p (float): probability of Gaussian blur
    '''

    def __init__(self, p):
        self.p = p
    
    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization:
    '''Solarization as a callable object. Apply solarization to an input image.

    Args:
        img (Image): an image in the PIL.Image format

    Returns:
        Image: a solarized image
    '''

    def __call__(self, img):
        return ImageOps.solarize((img))


class ContrastiveLearningViewGenerator(object):
    '''Data augmentation in PIS. Generate n patches for each image

    Args:
        num_var (int): number of patches for each image
    '''

    def __init__(self, is_eurosat=False, num_var=4):
        self.num_var = num_var
        self.is_eurosat = is_eurosat
    
    def __call__(self, x):
        if self.is_eurosat:
            random_resized_crop = transforms.RandomResizedCrop(224, scale=(0.25, 0.25), ratio=(1, 1))
        else:   # ['ucm', 'aid', 'pn', 'nr']
            random_resized_crop = transforms.RandomResizedCrop(224, scale=(0.25, 0.25), ratio=(1, 1),
                                                               interpolation=InterpolationMode.BICUBIC)
        
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
        aug_transforms = transforms.Compose([random_resized_crop,
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
                                             transforms.RandomGrayscale(p=0.2),
                                             GBlur(p=0.1),
                                             transforms.RandomApply([Solarization()], p=0.1),
                                             transforms.ToTensor(),
                                             normalize])
        
        augmented_x = [aug_transforms(x) for i in range(self.num_var)]

        return augmented_x
