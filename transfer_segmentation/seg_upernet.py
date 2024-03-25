import sys
sys.path.append('..')

import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

from func import compute_iou
from seg_datasets import get_dlrsd_split, get_potsdam_split, get_loveda_split


'''arguments for finetuning'''
parser = argparse.ArgumentParser(description='Segmentation downstream task for finetuning')

# backbone
parser.add_argument('--arch', type=str, default='swin_b',
                    help='backbone for segmentation finetuing')

# training arguments
parser.add_argument('--bs', type=int, default=10,
                    help='batch_size for segmentation finetuing')
parser.add_argument('--lr', type=float, default=0.003,
                    help='learning rate for segmentation finetuing')
parser.add_argument('--epoch', type=int, default=100,
                    help='max number of epochs to finish')

# arguments for dataset
parser.add_argument('--data', type=str, default='potsdam',
                    help='data name for segmentation finetuing')
parser.add_argument('--tr', type=float, default=0.01,
                    help='training ratio for the segmentation dataset')

# arguments for loading model path
parser.add_argument('--model_path', type=str, default="",
                    help='pretrained model directory')

args = parser.parse_args()
print(args)

if args.data == "ucm":
    num_classes = 18
    get_dataset = get_dlrsd_split
elif args.data == 'potsdam':
    # num_classes = 7
    num_classes = 6
    get_dataset = get_potsdam_split
elif args.data == 'loveda':
    num_classes = 8
    get_dataset = get_loveda_split
else:
    raise NotImplementedError('dataset {0} not implemented'.format(args.data))

print('number of claases is {0}'.format(num_classes))


def seg_finetune(net, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    for epoch in range(args.epoch):
        for images, labels in tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            outputs = net(images)['logits']
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        scheduler.step()

        print(f"Epoch {epoch}\tLoss {loss.item()}")
    
    # test for the last epoch
    miou = 0
    for counter, (images, labels) in enumerate(tqdm(test_loader)):
        images, labels = images.cuda(), labels.cuda()

        logits = net(images)['logits']
        preds = torch.argmax(logits, dim=1)
        miou += compute_iou(preds, labels, num_classes)

    miou /= (counter + 1)

    print('miou is: {0}'.format(miou))


if __name__ == '__main__':
    # prepare dataset
    train_dataset, test_dataset = get_dataset(train_ratio=args.tr)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)

    # prepare model
    backbone_config = SwinConfig(embed_dim=128,
                                 depths=[2, 2, 18, 2],
                                 num_heads=[4, 8, 16, 32],
                                 out_features=["stage1", "stage2", "stage3", "stage4"])

    config = UperNetConfig(backbone_config=backbone_config, use_auxiliary_head=False)
    config.num_labels = num_classes
    net = UperNetForSemanticSegmentation(config)

    # load pretrained model
    my_dict = torch.load(args.model_path)
    for i in list(my_dict.keys()):
        if not i.startswith('module.backbone.'):
            del my_dict[i]

    for i in list(my_dict.keys()):
        if i.endswith('mask'):
            del my_dict[i]
    
    state_dict = net.state_dict()
    for k in list(state_dict.keys()):
        if not k.startswith('backbone'):
            del state_dict[k]
    
    store_hf = list(state_dict.keys())
    new_state_dict = {}

    i = 0
    count = 0
    for key in list(my_dict.keys()):
        if 'qkv' in key:
            count += 1
            q, k, v = my_dict[key].chunk(3, dim=0)
            new_state_dict[store_hf[i]] = q
            new_state_dict[store_hf[i + 2]] = k
            new_state_dict[store_hf[i + 4]] = v
            if count % 2 == 0:
                i += 5
            else:
                i += 1
        elif 'downsample.norm.weight' in key:
            new_state_dict[store_hf[i+1]] = my_dict[key]
            i += 1
        elif 'downsample.norm.bias' in key:
            new_state_dict[store_hf[i+1]] = my_dict[key]
            i += 1
        elif 'downsample.reduction.weight' in key:
            new_state_dict[store_hf[i-2]] = my_dict[key]
            i += 1
        elif 'module.backbone.norm.weight' == key:
            new_state_dict[store_hf[-2]] = my_dict[key]
            i += 1
        elif 'module.backbone.norm.bias' == key:
            new_state_dict[store_hf[-1]] = my_dict[key]
            i += 1
        else:
            new_state_dict[store_hf[i]] = my_dict[key]
            i += 1
    
    print(net.load_state_dict(new_state_dict, strict=False))
    net.cuda()

    # segmentation finetuning
    seg_finetune(net, train_dataloader, test_dataloader)
