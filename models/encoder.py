import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_cnn_backbone, get_swint_backbone


class Encoder(nn.Module):
    '''encoder for pre-training

    Args:
        z_dim (int, optional): final output dimension. Defaults to 1024.
        hidden_dim (int, optional): hidden layer dimension. Defaults to 4096.
        norm_p (int, optional): the exponent value in the norm formulation. Defaults to 2.
        arch (str, optional): backbone. Defaults to 'resnet50'.
    '''
    def __init__(self, z_dim=1024, hidden_dim=4096, norm_p=2, arch='resnet50'):
        super().__init__()
        self.norm_p = norm_p

        if arch in ['resnet50', 'vgg16', 'res2net50']:
            self.backbone, feature_dim = get_cnn_backbone(arch)
        elif arch in ['swin_t', 'swin_s', 'swin_b', 'swin_l']:
            self.backbone, feature_dim = get_swint_backbone(arch)
        else:
            raise NotImplementedError('backbone {0} is not supported in this implementation'.format(arch))

        self.predictor = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU())
        self.projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, z_dim))

        # two learnable weights
        # self.var_sim_weight = nn.Parameter(torch.tensor(200.0))
        # self.tcr_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, is_test=False):
        feature = self.backbone(x)
        z_pre = self.predictor(feature)
        z = F.normalize(self.projector(z_pre), p=self.norm_p)

        if is_test:
            return z, z_pre
        else:
            return z


class FinetuneEncoder(nn.Module):
    '''Encoder for classification finetuning.

    Args:
        hidden_dim (int, optional): hidden layer dimension. Defaults to 4096.
        test_patches (int, optional): number of patches for each testing image
        num_classes (int, optional): number of classes for dataset
        arch (str, optional): backbone. Defaults to 'resnet50'.
    '''
    def __init__(self, hidden_dim=4096, test_patches=4, num_classes=21, arch='resnet50'):
        super().__init__()
        self.test_patches = test_patches

        if arch in ['resnet50', 'vgg16', 'res2net50']:
            self.backbone, feature_dim = get_cnn_backbone(arch)
        elif arch in ['swin_t', 'swin_s', 'swin_b', 'swin_l']:
            self.backbone, feature_dim = get_swint_backbone(arch)
        else:
            raise NotImplementedError('backbone {0} is not supported in this implementation'.format(arch))
        
        self.predictor = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU())
        self.chunk_avg = Chunk_Avg(num_chunks=self.test_patches, normalize=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        feature = self.backbone(x)
        feature = self.predictor(feature)
        feature = self.chunk_avg(feature)
        out = self.fc(feature)
        return out


class Chunk_Avg(nn.Module):
    '''Average the features of patches. A module in the encoder.

    Args:
        num_chunk (int): equal to num_patches
        normalize: (bool)
    '''

    def __init__(self, num_chunks=4, normalize=False):
        super().__init__()
        self.num_chunks = num_chunks
        self.normalie = normalize
    
    def forward(self, x):
        x_list = x.chunk(self.num_chunks, dim=0)
        x = torch.stack(x_list, dim=0)
        if not self.normalie:
            return x.mean(0)
        else:
            return F.normalize(x.mean(0), dim=1)
