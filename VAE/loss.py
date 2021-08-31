import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



def VAE_loss(x, x_reconst, mu, log_var):
    reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False, reduce=False)
    print(reconst_loss)
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = reconst_loss+kl_div
    out = {'reconst_loss':reconst_loss, 'kl_div':kl_div, 'loss': loss}
    return out
