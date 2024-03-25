from cProfile import label
import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion, get_split_loaders,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
import os
import logging
import json
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
from torch import nn, optim
from torch.optim import lr_scheduler

def get_scheduler(optimizer, opt, lr_policy):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(opt.epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        step_size = opt.epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

opt.batch_size = 8
opt.learning_rate = 2e-4
opt.loss_function = 'bce'

save_path = 'tmp'+ '/' + opt.dataset + '_' + opt.backbone + '_'+ opt.mode
change_path = 'tmp'+ '/' + opt.dataset + '_' + opt.backbone + '_'+ opt.mode + '/change'
label_path = 'tmp'+ '/' + opt.dataset + '_' + opt.backbone + '_'+ opt.mode + '/label'

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter('.tmp/log/' + opt.dataset + '_' + opt.backbone + '_'+ opt.mode + '_' + f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

"""
Set up environment: define paths, download data, and set device
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)

if opt.dataset == 'cdd':
    opt.dataset_dir = '/home/anxiao/Datasets/CDD/Real/subset/'
elif opt.dataset == 'levir':
	opt.dataset_dir = '/home/anxiao/Datasets/LEVIR/'


# train_loader, val_loader = get_loaders(opt)
train_loader, test_loader = get_split_loaders(opt)


"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = load_model(opt, dev)

criterion = get_criterion(opt)

# todo modify the optimizer
if opt.backbone == 'resnet':
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=0.0005) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    # scheduler = get_scheduler(optimizer, opt, 'linear')
else:
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
    # scheduler = get_scheduler(optimizer, opt, 'linear')
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1

for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    """
    Begin Training
    """
    model.train()
    # logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels, fname in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1
        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop
        cd_preds = model(batch_img1, batch_img2)

        cd_loss = criterion(cd_preds, labels)
        loss = cd_loss
        loss.backward()
        optimizer.step()

        #cd_preds = cd_preds[-1] # BIT输出不是tuple
        _, cd_preds = torch.max(cd_preds, 1)

        # Calculate and log other batch metrics
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))

        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               pos_label=1,zero_division=0)

        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_last_lr())

        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

"""
Begin Validation
"""
model.eval()
with torch.no_grad():
    for batch_img1, batch_img2, labels, fname in tqdm(test_loader):
        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # Get predictions and calculate loss
        cd_preds = model(batch_img1, batch_img2)

        cd_loss = criterion(cd_preds, labels)

        #cd_preds = cd_preds[-1] # BIT输出不是tuple
        _, cd_preds = torch.max(cd_preds, 1)

        # Calculate and log other batch metrics
        cd_corrects = (100 *
                        (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                        (labels.size()[0] * (opt.patch_size**2)))

        cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                            cd_preds.data.cpu().numpy().flatten(),
                            average='binary',
                            pos_label=1,zero_division=0)

        val_metrics = set_metrics(val_metrics,
                                cd_loss,
                                cd_corrects,
                                cd_val_report,
                                scheduler.get_last_lr())

        # log the batch mean metrics
        mean_val_metrics = get_mean_metrics(val_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'val': v}, total_step)
        
        prediction_np = cd_preds.data.cpu().numpy()
        label_np = labels.data.cpu().numpy()
        fname_np = fname
        '''
        for i in range(len(fname_np)):
            plt.figure()
            plt.imshow(prediction_np[i], cmap='gray')
            plt.savefig(os.path.join(change_path, fname_np[i]))
            plt.close()
            plt.figure()
            plt.imshow(label_np[i], cmap='gray')
            plt.savefig(os.path.join(label_path, fname_np[i]))
            plt.close()
        '''
        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

writer.close()  # close tensor board
print('Done!')