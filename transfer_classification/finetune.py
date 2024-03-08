import sys
sys.path.append('..')

import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from func import accuracy
from dataset.datasets import load_few_dataset
from models.encoder import FinetuneEncoder


'''arguments for finetuning'''
parser = argparse.ArgumentParser(description='Few shot training and testing using Linear layer for finetuning')

# backbone
parser.add_argument('--arch', type=str, default='resnet50',
                    help='backbone for finetuning')

# training arguments
parser.add_argument('--bs', type=int, default=32,
                    help='batch_size for finetuning')
parser.add_argument('--lr', type=float, default=0.03,
                    help='learning rate for finetuning')
parser.add_argument('--epoch', type=int, default=100,
                    help='max number of epochs to finish')

# arguments for dataset
parser.add_argument('--data', type=str, default='ucm',
                    help='data name for finetuning')
parser.add_argument('--num_var', type=int, default=16,
                    help='number of patches for finetuning')
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


def finetuning(net, train_loader, test_loader):
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    for epoch in range(args.epoch):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
            net.zero_grad()
            optimizer.zero_grad()

            x_batch = torch.cat(x_batch, dim=0).cuda()
            y_batch = y_batch.cuda()
            
            logits = net(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            loss.backward()
            optimizer.step()
        scheduler.step()

        top1_train_accuracy /= (counter + 1)

        print(f'Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}, Loss {loss.item()}')
        '''
        # val after each epoch of training
        if epoch >= 50:
            with torch.no_grad():
                top1_accuracy = 0
                top5_accuracy = 0
                for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
                    x_batch = torch.cat(x_batch, dim=0).cuda()
                    y_batch = y_batch.cuda()

                    logits = net(x_batch)

                    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
                    top1_accuracy += top1[0]
                    top5_accuracy += top5[0]

                    if counter == 19:
                        break

                top1_accuracy /= (counter + 1)
                top5_accuracy /= (counter + 1)

                print(f'Top1 Val accuracy: {top1_accuracy.item()}\tTop5 Val acc: {top5_accuracy.item()}')

                if top1_accuracy > 70.0:
                    top1_accuracy = 0
                    top5_accuracy = 0
                    for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
                        x_batch = torch.cat(x_batch, dim=0).cuda()
                        y_batch = y_batch.cuda()

                        logits = net(x_batch)

                        top1, top5 = accuracy(logits, y_batch, topk=(1,5))
                        top1_accuracy += top1[0]
                        top5_accuracy += top5[0]

                    top1_accuracy /= (counter + 1)
                    top5_accuracy /= (counter + 1)

                    print(f'Top1 Val accuracy: {top1_accuracy.item()}\tTop5 Val acc: {top5_accuracy.item()}')
            '''
    # test after the last epoch of training
    with torch.no_grad():
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
            x_batch = torch.cat(x_batch, dim=0).cuda()
            y_batch = y_batch.cuda()

            logits = net(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

        print(f'Top1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}')


if __name__ == '__main__':
    # prepare dataset
    train_dataset, test_dataset = load_few_dataset(data_name=args.data, num_sample=args.num_samples, num_var=args.num_var)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=4)

    # prepare model
    # net = finetune_encoder(test_patches=args.num_var, arch = args.arch, num_classes=num_classes)
    net = FinetuneEncoder(test_patches=args.num_var, arch = args.arch, num_classes=num_classes)
    net = nn.DataParallel(net)

    # load pretrained model
    state_dict = torch.load(args.model_path)
    for k in list(state_dict.keys()):
        if k.startswith('module.projector.'):
            del state_dict[k]

    print(net.load_state_dict(state_dict, strict=False))
    net.cuda()

    # finetuning
    finetuning(net, train_dataloader, test_dataloader)
