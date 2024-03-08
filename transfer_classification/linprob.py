import sys
sys.path.append('..')

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.encoder import Encoder
from func import chunk_avg, linear
from dataset.datasets import load_few_dataset


'''arguments for linear probing'''
parser = argparse.ArgumentParser(description='Few shot training and testing using Linear layer for linear probing')

# backbone
parser.add_argument('--arch', type=str, default='resnet50',
                    help='backbone for linear probing')

# training arguments
parser.add_argument('--bs', type=int, default=32,
                    help='batch_size for linear probing')
parser.add_argument('--lr', type=float, default=0.03,
                    help='learning rate for linear probing')
parser.add_argument('--epoch', type=int, default=100,
                    help='max number of epochs to finish')

# arguments for dataset
parser.add_argument('--data', type=str, default='ucm',
                    help='data name for linear probing')
parser.add_argument('--num_var', type=int, default=16,
                    help='number of patches for linear probing')
parser.add_argument('--num_samples', type=int, default=5,
                    help='number of samples for few-shot training')

# arguments for loading model path
parser.add_argument('--model_path', type=str, default="",
                    help='pretrained model directory')

args = parser.parse_args()
print(args)

# define number of classes
if args.data == "ucm":
        num_classes = 21
elif args.data == "aid":
    num_classes = 30
elif args.data == 'nr':
    num_classes = 45
elif args.data == 'pn':
    num_classes = 38
elif args.data == 'rsd':
    num_classes = 46
elif args.data == 'eurosat':
    num_classes = 10
else:
     raise NotImplementedError('dataset {0} not implemented'.format(args.data))


def linear_probing(net, train_dataloader, test_dataloader):
    train_z_full_list, train_y_list, test_z_full_list, test_y_list = [], [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(train_dataloader):
            x = torch.cat(x, dim=0).cuda()
            
            z, z_pre = net(x, is_test=True)

            z_pre = chunk_avg(z_pre, args.num_var)
            z_pre = z_pre.detach().cpu()
            
            train_z_full_list.append(z_pre)
            train_y_list.append(y)
                
        for x, y in tqdm(test_dataloader):
            x = torch.cat(x, dim = 0).cuda()
            
            z, z_pre = net(x, is_test=True)

            z_pre = chunk_avg(z_pre, args.num_var)
            z_pre = z_pre.detach().cpu()
           
            test_z_full_list.append(z_pre)
            test_y_list.append(y)
                
    train_features_full, train_labels = torch.cat(train_z_full_list, dim=0), torch.cat(train_y_list, dim=0)
    test_features_full, test_labels = torch.cat(test_z_full_list, dim=0), torch.cat(test_y_list, dim=0)

    print("Using Linear Probing to evaluate accuracy")
    linear(train_features_full, train_labels, test_features_full, test_labels, bs=args.bs, epoch=args.epoch, lr=args.lr, num_classes = num_classes)


if __name__ == '__main__':
    # prepare dataset
    memory_dataset, test_dataset = load_few_dataset(data_name=args.data, num_sample=args.num_samples, num_var=args.num_var)
    memory_dataloader = DataLoader(memory_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)

    # prepare model
    net = Encoder(args.arch)
    net = nn.DataParallel(net)

    # load pretrained model
    state_dict = torch.load(args.model_path)
    net.load_state_dict(state_dict, strict=True)
    net.cuda()
    net.eval()

    # linear probing
    linear_probing(net, memory_dataloader, test_dataloader)
