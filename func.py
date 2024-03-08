import os

import torch
from torch import nn, optim
from torch.utils import data
import torch.nn.functional as F
from torch.utils.data import DataLoader


def chunk_avg(x, n_chunks=4, normalize=False):
    '''Compute the average of the features by chunking it into n_chunks and averaging them.
    n_chunks is equal to the number of patches.

    Args:
        x (tensor): the combined features of the patches.
        n_chunks (int, optional): equal to the number of patches. Defaults to 4.
        normalize (bool, optional): normalization. Defaults to False.

    Returns:
        tensor: averaged features of the patches.
    '''
    x_list = x.chunk(n_chunks, dim=0)
    x = torch.stack(x_list, dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0), dim=1)

def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output: predictions
        target: ground truth labels
        topk (tuple, optional): (1, k). Defaults to (1,).

    Returns:
        float: top-1 (and top-k) accuracy
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_iou(pred, target, num_classes):
    '''compute MIoU for segmentation downstream tasks.

    Args:
        pred: predictions
        target: ground truth labels
        num_classes (int): the number of classes for the segmentation task.

    Returns:
        float: mIoU
    '''
    iou = 0
    for i in range(1, num_classes): # exclude background
        pred_idx = (pred == i)
        target_idx = (target == i)
        intersection = torch.logical_and(pred_idx, target_idx).sum()
        union = torch.logical_or(pred_idx, target_idx).sum()
        iou += float(intersection) / (float(union) + 1e-10)
    iou = iou / (num_classes - 1)
    return iou

def linear(train_features, train_labels, test_features, test_labels, bs=32, epoch=100, lr=0.0075, num_classes=21):    
    train_data = tensor_dataset(train_features,train_labels)
    test_data = tensor_dataset(test_features,test_labels)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=True, drop_last=False, num_workers=2)
    
    LL = nn.Linear(train_features.shape[1],num_classes)
    LL = LL.cuda()
    optimizer = torch.optim.SGD(LL.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch)
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    test_acc_list = []
    for epoch in range(epoch):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            
            logits = LL(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step() 

        top1_train_accuracy /= (counter + 1)

        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            logits = LL(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        
        test_acc_list.append(top1_accuracy)
        
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    acc_vect = torch.tensor(test_acc_list)
    print('best linear test acc {}, last acc {}'.format(acc_vect.max().item(),acc_vect[-1].item()))

            
class tensor_dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = x.shape[0]
    
    def __getitem__(self, indx):
        return self.x[indx], self.y[indx]
    
    def __len__(self):
        return self.length


def save_model(epoch, net, model_dir):
    '''save the model

    Args:
        epoch (int): current epoch
        net (nn.Module): the model
        model_dir (str): the directory to save the model
    '''
    # if (epoch == 1) or (epoch % 5 == 0):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(net.state_dict(), os.path.join(model_dir, 'epoch_{0}.pt'.format(epoch)))
