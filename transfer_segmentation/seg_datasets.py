import torch
import torchvision
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image


class Potsdam(Dataset):
    def __init__(self, img_root='./data/img', label_root='./data/label'):
        self.img_root = img_root
        self.label_root = label_root

        imgs = os.listdir(img_root)
        imgs.sort()
        labels = os.listdir(label_root)
        labels.sort()

        image_files = [i for i in imgs if i.endswith('.png')]
        label_files = [l for l in labels if l.endswith('.png')]
        self.dataset = [(os.path.join(img_root, image_files[i]), os.path.join(label_root, label_files[i])) for i in range(len(image_files))]

    def __getitem__(self, idx):
        img_path, label_path = self.dataset[idx]
        img = Image.open(img_path)
        label = Image.open(label_path)
        img, label = random_crop(img, label, 224, 224)

        img = torchvision.transforms.ToTensor()(img)
        label = rgb2gray(label, 'potsdam')
        label = torch.from_numpy(label).long()
        return img, label
    
    def __len__(self):
        return len(self.dataset)


def get_potsdam_split(train_ratio=0.2):
    '''randomly split potsdam dataset into trainset and testset with the specific ratio'''
    potsdam_dataset = Potsdam(img_root='/home/anxiao/Datasets/Potsdam/2_Ortho_RGB_256',
                              label_root='/home/anxiao/Datasets/Potsdam/5_Labels_all_noBoundary_256')
    
    train_size = int(train_ratio * len(potsdam_dataset))
    test_size = len(potsdam_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(potsdam_dataset, [train_size, test_size])
    
    return trainset, testset


class DLRSD(Dataset):
    def __init__(self, img_root='./data/img', label_root='./data/label'):
        self.img_root = img_root
        self.label_root = label_root
        self.data_img = torchvision.datasets.ImageFolder(root=self.img_root)
        self.data_label = torchvision.datasets.ImageFolder(root=self.label_root)
        self.dataset = [(self.data_img.samples[i][0], self.data_label.samples[i][0]) for i in range(len(self.data_img))]

    def __getitem__(self, idx):
        img_path, label_path = self.dataset[idx]
        img = Image.open(img_path)
        label = Image.open(label_path)
        img, label = random_crop(img, label, 224, 224)

        img = torchvision.transforms.ToTensor()(img)
        label = torch.from_numpy(np.array(label)).long()
        return img, label
    
    def __len__(self):
        return len(self.dataset)


def get_dlrsd_split(train_ratio=0.2):
    '''randomly split dlrsd dataset into trainset and testset with the specific ratio'''
    dlrsd_dataset = DLRSD(img_root='/home/anxiao/Datasets/UCMerced_LandUse/Images',
                          label_root='/home/anxiao/Datasets/DLRSD/Images')
    
    train_size = int(train_ratio * len(dlrsd_dataset))
    test_size = len(dlrsd_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dlrsd_dataset, [train_size, test_size])
    
    return trainset, testset


class LoveDA(Dataset):
    def __init__(self, img_root='./data/img', label_root='./data/label'):
        self.img_root = img_root
        self.label_root = label_root
        self.data_img = torchvision.datasets.ImageFolder(root=self.img_root)
        self.data_label = torchvision.datasets.ImageFolder(root=self.label_root)
        self.dataset = [(self.data_img.samples[i][0], self.data_label.samples[i][0]) for i in range(len(self.data_img))]

    def __getitem__(self, idx):
        img_path, label_path = self.dataset[idx]
        img = Image.open(img_path)
        label = Image.open(label_path)
        img, label = random_crop(img, label, 224, 224)

        img = torchvision.transforms.ToTensor()(img)
        label = torch.from_numpy(np.array(label)).long()
        return img, label
    
    def __len__(self):
        return len(self.dataset)


def get_loveda_split(train_ratio=0.2):
    '''randomly split dlrsd dataset into trainset and testset with the specific ratio'''
    loveda_dataset = LoveDA(img_root='/home/anxiao/Datasets/LoveDA/Combine_256/Images',
                            label_root='/home/anxiao/Datasets/LoveDA/Combine_256/Labels')
    
    train_size = int(train_ratio * len(loveda_dataset))
    test_size = len(loveda_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(loveda_dataset, [train_size, test_size])
    
    return trainset, testset


def random_crop(image, label, height, width):
    '''random crop image and label with the same size'''
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, output_size=(height, width))
    image = torchvision.transforms.functional.crop(image, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return image, label

# color map for segmentation dataset
# Potsdam_COLORMAP = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]])
Potsdam_COLORMAP = np.array([[255, 0, 0], [255, 255, 255], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]])

def rgb2gray(label, data_name):
    '''convert rgb label to gray label

    Args:
        label: ground truth label
        data_name (string): segmentation dataset name

    Returns:
        array: label mask with gray values
    '''
    if data_name == 'potsdam':
        COLORMAP = Potsdam_COLORMAP
    else:
        raise NotImplementedError

    label = np.array(label)
    label_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.int8)
    for i, color in enumerate(COLORMAP):
        locations = np.all(label==color, axis=-1)
        label_mask[locations] = i
    return label_mask.astype(np.uint8)
