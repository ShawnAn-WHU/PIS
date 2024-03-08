import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity_Loss(nn.Module):
    '''compute the similarity loss between the average of the patches and each patch.
       the purpose is to cluster each patch to the average of the patches.
    '''
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        num_var = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_var):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim / num_var
        z_sim_out = z_sim.clone().detach()

        return -z_sim, z_sim_out


class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)
    

def cal_TCR(z, criterion, num_var):
    '''compute TCR loss for each patch and average them

    Args:
        z (tensor): the output combined features
        criterion: the loss function for TCR
        num_var (int): the number of patches
    '''
    z_list = z.chunk(num_var, dim=0)
    loss = 0
    for i in range(num_var):
        loss += criterion(z_list[i])
    loss = loss / num_var
    return loss
