import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def VAE_loss(x, x_reconst, mu, log_var):
    reconst_loss = F.mse_loss(x_reconst, x, reduce=False) # [N x D] # l2?
    reconst_loss = torch.sum(reconst_loss,axis=-1) # [N]
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=-1) # [N]
    loss = reconst_loss#+0.01*kl_div
    out = {'reconst_loss':reconst_loss, 'kl_div':kl_div, 'loss': loss}
    return out

def VAE_eval(x, x_reconst, mu, log_var):
    TEMP = 10 # Temperature
    x = F.sigmoid(x/TEMP)
    x_reconst = F.sigmoid(x_reconst/TEMP)
    reconst_loss = F.mse_loss(x_reconst, x, reduce=False) # [N x D] # l2?
    reconst_loss = torch.mean(reconst_loss,axis=-1) # [N]
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=-1) # [N]
    loss = reconst_loss+kl_div
    out = {'reconst_loss':reconst_loss, 'kl_div':kl_div, 'loss': loss}
    return out