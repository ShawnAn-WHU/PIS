import wandb
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from lars import LARSWrapper
from models.encoder import Encoder
from func import chunk_avg, save_model
import torch.optim.lr_scheduler as lr_scheduler
from dataset.datasets import SSL4EO_RGB, SSL4EO_RGB_MIX
from loss import cal_TCR, Similarity_Loss, TotalCodingRate


'''arguments for pre-training'''
parser = argparse.ArgumentParser(description='PIS pretraining')

# backbone
parser.add_argument('--arch', type=str, default='resnet50',
                    help='backbone for pre-training')

# pretraining arguments
parser.add_argument('--bs', type=int, default=32,
                    help='batch_size for pre-training')
parser.add_argument('--lr', type=float, default=0.3,
                    help='learning rate')
parser.add_argument('--epoch', type=int, default=30,
                    help='max number of epochs to finish')

# arguments for dataset and loss function
parser.add_argument('--data', type=str, default='SSL4EO_RGB_MIX',
                    help='data name')
parser.add_argument('--num_var', type=int, default=16,
                    help='number of patches used in PIS')
parser.add_argument('--eps', type=float, default=0.2,
                    help='eps for TCR')
parser.add_argument('--tcr', type=float, default=1,
                    help='coefficient of tcr')
parser.add_argument('--var_sim', type=int, default=200,
                    help='coefficient of cosine similarity')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')

args = parser.parse_args()
print(args)

# directory for logs
wandb.login()
wandb.init(project='PIS', name=f'pretrain_{args.arch}_bs{args.bs}_lr{args.lr}_np{args.num_var}')
dir_name = f'./logs/{args.arch}_bs{args.bs}_lr{args.lr}_np{args.num_var}_epoch{args.epoch}'

# prepare dataset
torch.multiprocessing.set_sharing_strategy('file_system')

if args.data == 'SSL4EO_RGB':
    train_dataset = SSL4EO_RGB(csv_path='dataset/SSL4EO_RGB.csv', num_var=args.num_var)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
elif args.data == 'SSL4EO_RGB_MIX':
    train_dataset = SSL4EO_RGB_MIX(csv_path='dataset/SSL4EO_RGB_MIX.csv', num_var=int(args.num_var/4))
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
else:
    raise NotImplementedError('dataset {0} is not implemented'.format(args.data))

# prepare model
net = Encoder(arch=args.arch)
net = nn.parallel.DataParallel(net)
net = net.cuda()

# optimizer
if args.arch in ['resnet50', 'vgg16', 'res2net50']:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
elif args.arch in ['swin_t', 'swin_s', 'swin_b', 'swin_l']:
    # optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)

# optimizer = LARSWrapper(optimizer, eta=0.005, clip=True, exclude_bias_n_norm=True)

# scheduler
num_converge = (251079 // args.bs) * args.epoch
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_converge, eta_min=0, last_epoch=-1)

# loss function
contrastive_loss = Similarity_Loss().cuda()
criterion = TotalCodingRate(eps=args.eps).cuda()

# grad scaler
scaler = torch.cuda.amp.GradScaler()


def main():
    for epoch in range(1, args.epoch + 1):
        for iter, data in enumerate(tqdm(train_dataloader, leave=False)):
            net.zero_grad()
            optimizer.zero_grad()

            data = torch.cat(data, dim=0)
            data = data.cuda()

            with torch.cuda.amp.autocast():
                z = net(data)

                z_list = z.chunk(args.num_var, dim=0)
                z_avg = chunk_avg(z, n_chunks=args.num_var)

                # compute loss
                loss_contrast, _ = contrastive_loss(z_list, z_avg)
                loss_TCR = cal_TCR(z, criterion, args.num_var)
                loss = args.var_sim * loss_contrast + args.tcr * loss_TCR

                # weight_1 = nn.functional.softplus(net.module.var_sim_weight)
                # weight_2 = nn.functional.softplus(net.module.tcr_weight)
                # loss = weight_1 * loss_contrast + weight_2 * loss_TCR
                
            
            # update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            '''
            loss.backward()
            optimizer.step()
            scheduler.step()
            '''

        # print loss and lr
        print('At epoch', epoch, 'loss similarity is:', loss_contrast.item(), 'loss TCR is:', loss_TCR.item())
        # print('At epoch', epoch, 'weight_1 is:', weight_1.item(), 'weight_2 is:', weight_2.item())
        
        # log loss and lr with wandb
        wandb.log({'loss_contrast': loss_contrast.item(),
                   'loss_TCR': loss_TCR.item(),
                   'loss': loss.item(),
                   'learning_rate': optimizer.param_groups[0]['lr']
                  }, step=epoch)
        
        # save model
        save_model(epoch, net, dir_name)


if __name__ == '__main__':
    main()
