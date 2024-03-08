import os
import csv
import random
from PIL import Image

import torchvision
from torch.utils.data import Subset
from torch.utils.data import Dataset

from .aug import ContrastiveLearningViewGenerator


class SSL4EO_RGB(Dataset):
    '''RGB only. Using only one season (e.g. spring).

    Args:
        csv_path (path): .csv path of SSL4EO dataset. 'SSL4EO_RGB.csv'
        num_var (int): number of patches for each image
    '''

    def __init__(self, csv_path, num_var=4):
        super().__init__()
        self.csv_path = csv_path
        self.transform = ContrastiveLearningViewGenerator(num_var = num_var)
        self.img_paths = self.read_csv(self.csv_path)

    def read_csv(self, csv_path):
        img_paths = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if (i % 4) == 0:    # using spring only
                    img_paths.append(row)
        return img_paths
    
    def __getitem__(self, item):
        img = Image.open(self.img_paths[item][0])
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_paths)


class SSL4EO_RGB_MIX(Dataset):
    '''RGB only. Using four seasons (take seasonal pairs as positive)

    Args:
        csv_path (path): .csv path of SSL4EO dataset. 'SSL4EO_RGB_MIX.csv'
        num_var (int): number of patches for each image
    '''

    def __init__(self, csv_path, num_var=4):
        super().__init__()
        self.csv_path = csv_path
        self.transform = ContrastiveLearningViewGenerator(num_var = num_var)
        self.img_paths = self.read_csv(self.csv_path)

    def read_csv(self, csv_path):
        img_paths = []
        seasons = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for i in range(len(row)):
                    seasons.append(row[i])
                img_paths.append(seasons)
                seasons = []
        return img_paths
    
    def __getitem__(self, item):
        imgs = self.img_paths[item]
        image_spring = Image.open(imgs[0])
        image_summer = Image.open(imgs[1])
        image_autumn = Image.open(imgs[2])
        image_winter = Image.open(imgs[3])
        if self.transform is not None:
            image_spring = self.transform(image_spring)
            image_summer = self.transform(image_summer)
            image_autumn = self.transform(image_autumn)
            image_winter = self.transform(image_winter)
        image_all_seasons = image_spring + image_summer + image_autumn + image_winter
        return image_all_seasons
    
    def __len__(self):
        return len(self.img_paths)


def load_few_dataset(data_name, num_sample=5, num_var=4, data_root='/home/anxiao/Datasets'):
    '''Load training and testing dataset for few-shot in fine-tuning.

    Args:
        data_name (str): name of dataset
        num_sample (int, optional): number of samples for each category. Defaults to 5.
        num_var (int, optional): number of patches for each image. Defaults to 4.
        data_root (str, optional): path to the 'data_name' dataset. Defaults to './data'.

    Returns:
        Dataset: training and testing dataset for few-shot.
    '''
    _name = data_name.lower()
    is_eurosat = (_name == 'eurosat')

    transform = ContrastiveLearningViewGenerator(is_eurosat, num_var = num_var)

    if _name == 'ucm':
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'UCMerced_LandUse/Images'), transform=transform)
    elif _name == 'aid':
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'AID'), transform=transform)
    elif _name == 'pn':
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'PatternNet/images'), transform=transform)
    elif _name == 'nr':
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'NWPU-RESISC45'), transform=transform)
    elif _name == 'eurosat':
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'EuroSAT'), transform=transform)
    
    class_names = dataset.classes

    class_indices = {class_name: [] for class_name in class_names}
    for idx, (_, class_idx) in enumerate(dataset.samples):
        class_indices[class_names[class_idx]].append(idx)

    train_indices = []
    test_indices = []
    for class_name, indices in class_indices.items():
        train_indices.extend(random.sample(indices, num_sample))  # choose samples for few-shot training
        test_indices.extend(idx for idx in indices if idx not in train_indices)  # the rest are for testing

    few_shot_train_dataset = Subset(dataset, train_indices)
    few_shot_test_dataset = Subset(dataset, test_indices)
    return few_shot_train_dataset, few_shot_test_dataset
