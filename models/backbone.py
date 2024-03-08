import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models import swin_transformer


def get_cnn_backbone(arch='resnet50'):
    '''get the CNN backbone of the encoder.

    Args:
        arch (string): architecture of the CNN backbone (default: 'resnet50')
    '''
    if arch == 'resnet50':
        backbone = models.resnet50()
        backbone.fc = nn.Identity()
        feature_dim = 2048
        return backbone, feature_dim
    elif arch == 'vgg16':
        backbone = models.vgg16_bn()
        backbone.classifier._modules['6'] = torch.nn.Identity()
        feature_dim = 4096
        return backbone, feature_dim
    elif arch == 'res2net50':
        backbone = timm.models.res2net.res2net50_26w_4s()
        backbone.fc = nn.Identity()
        feature_dim = 2048
        return backbone, feature_dim
    else:
        raise NotImplementedError('CNN backbone {0} is not supported in this implementation'.format(arch))

def get_swint_backbone(arch='swint_b', num_classes=0):
    '''get the Swin Transformer backbone of the encoder.

    Args:
        arch (string): architecture of the Swin Transformer backbone (default: 'swint_b')
        num_classes (int, optional): number of categories. Defaults to 0 for pretraining.
    '''
    if arch == 'swin_t':
        backbone = swin_transformer.swin_tiny_patch4_window7_224(num_classes=num_classes)
        feature_dim = 768
        return backbone, feature_dim
    elif arch == 'swin_s':
        backbone = swin_transformer.swin_small_patch4_window7_224(num_classes=num_classes)
        feature_dim = 768
        return backbone, feature_dim
    elif arch == 'swin_b':
        backbone = swin_transformer.swin_base_patch4_window7_224(num_classes=num_classes)
        feature_dim = 1024
        return backbone, feature_dim
    elif arch == 'swin_l':
        backbone = swin_transformer.swin_large_patch4_window7_224(num_classes=num_classes)
        feature_dim = 1536
        return backbone, feature_dim
    else:
        raise NotImplementedError('Swin Transformer backbone {0} is not supported in this implementation'.format(arch))
