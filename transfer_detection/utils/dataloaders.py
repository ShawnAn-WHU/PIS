import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr
from torch.utils.data import Dataset
import torch


class CDD(Dataset):
    def __init__(self, data_root, aug=True):
        self.aug = aug

        train_name = [i for i in os.listdir(data_root + 'train/A/') if i.endswith('.jpg')]
        val_name = [i for i in os.listdir(data_root + 'val/A/') if i.endswith('.jpg')]
        test_name = [i for i in os.listdir(data_root + 'test/A/') if i.endswith('.jpg')]

        train_data_path = []
        val_data_path = []
        test_data_path = []

        for img in train_name:
            train_data_path.append([data_root + 'train/', img])
        for img in val_name:
            val_data_path.append([data_root + 'val/', img])
        for img in test_name:
            test_data_path.append([data_root + 'test/', img])
        
        self.all_data_path = train_data_path + val_data_path + test_data_path
        
    def __getitem__(self, idx):
        name = self.all_data_path[idx][1]
        img_path = self.all_data_path[idx]
        img1 = Image.open(img_path[0] + 'A/' + img_path[1])
        img2 = Image.open(img_path[0] + 'B/' + img_path[1])
        label = Image.open(img_path[0] + 'OUT/' + img_path[1])
        sample = {'image': (img1, img2), 'label': label}

        if self.aug:
            sample = tr.train_transforms(sample)
        else:
            sample = tr.test_transforms(sample)

        return sample['image'][0], sample['image'][1], sample['label'], name
    
    def __len__(self):
        return len(self.all_data_path)


def get_cdd_split(opt):
    '''randomly split potsdam dataset into trainset and testset with the specific ratio'''
    potsdam_dataset = CDD(data_root=opt.dataset_dir, aug=True)
    
    train_size = int(opt.tr * len(potsdam_dataset))
    test_size = len(potsdam_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(potsdam_dataset, [train_size, test_size])
    
    return trainset, testset


class LEVIR(Dataset):
    def __init__(self, data_root, aug=True):
        self.aug = aug

        train_name = [i for i in os.listdir(data_root + 'train_256/A/') if i.endswith('.png')]
        val_name = [i for i in os.listdir(data_root + 'val_256/A/') if i.endswith('.png')]
        test_name = [i for i in os.listdir(data_root + 'test_256/A/') if i.endswith('.png')]

        train_data_path = []
        val_data_path = []
        test_data_path = []

        for img in train_name:
            train_data_path.append([data_root + 'train_256/', img])
        for img in val_name:
            val_data_path.append([data_root + 'val_256/', img])
        for img in test_name:
            test_data_path.append([data_root + 'test_256/', img])
        
        self.all_data_path = train_data_path + val_data_path + test_data_path
        
    def __getitem__(self, idx):
        name = self.all_data_path[idx][1]
        img_path = self.all_data_path[idx]
        img1 = Image.open(img_path[0] + 'A/' + img_path[1])
        img2 = Image.open(img_path[0] + 'B/' + img_path[1])
        label = Image.open(img_path[0] + 'label/' + img_path[1])
        sample = {'image': (img1, img2), 'label': label}

        if self.aug:
            sample = tr.train_transforms(sample)
        else:
            sample = tr.test_transforms(sample)

        return sample['image'][0], sample['image'][1], sample['label'], name
    
    def __len__(self):
        return len(self.all_data_path)


def get_levir_split(opt):
    '''randomly split potsdam dataset into trainset and testset with the specific ratio'''
    levir_dataset = LEVIR(data_root=opt.dataset_dir, aug=True)
    
    train_size = int(opt.tr * len(levir_dataset))
    test_size = len(levir_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(levir_dataset, [train_size, test_size])
    
    return trainset, testset


'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/OUT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/OUT/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset

'''
Load all testing data paths
'''
def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + 'test/OUT/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    img1 = Image.open(dir + 'A/' + name)
    img2 = Image.open(dir + 'B/' + name)
    label = Image.open(label_path)
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label'], name


class CDDloader(data.Dataset):

    def __init__(self, full_load, flag = 'trn', aug=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

        print('load {} cdd {} pairs'.format(len(self.full_load), flag))

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
        
class LEVIRloader(data.Dataset):

    def __init__(self, full_load, flag = 'trn', aug=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

        print('load {} levir {} pairs'.format(len(self.full_load), flag))

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
