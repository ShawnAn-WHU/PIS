import sys
sys.path.append('..')

import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50

from func import compute_iou
from seg_datasets import get_dlrsd_split, get_potsdam_split, get_loveda_split


'''arguments for finetuning'''
parser = argparse.ArgumentParser(description='Segmentation downstream task for finetuning')

# backbone
parser.add_argument('--arch', type=str, default='fcn_resnet50',
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
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    for epoch in range(args.epoch):
        for images, labels in tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            outputs = net(images)['out']
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

        logits = net(images)['out']
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
    net = fcn_resnet50(weights=None, num_classes=num_classes)
    net = nn.DataParallel(net)

    # load pretrained model
    '''
    checkoint = torch.load(args.model_path)
    state_dict = checkoint['model_state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.pre_feature.') or k.startswith('module.projection.'):
            del state_dict[k]
    '''
    
    state_dict = torch.load(args.model_path)
    for k in list(state_dict.keys()):
        if k.startswith('module.projector.') or k.startswith('module.predictor.'):
            del state_dict[k]
    
    '''
    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint['state_dict']
    '''
    
    print(net.load_state_dict(state_dict, strict=False))
    net.cuda()

    # segmentation finetuning
    seg_finetune(net, train_dataloader, test_dataloader)
