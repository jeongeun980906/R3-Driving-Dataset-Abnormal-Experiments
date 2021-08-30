from MDN.network import LinActv,MixturesOfGaussianLayer
from MDN.mha.encoder import Encoder
import torch.nn as nn
from collections import OrderedDict
import torch

class MixtureDensityNetwork_MHA(nn.Module):
    def __init__(self,name='mdn',x_dim=1,y_dim=2,k=5,h_dim=128,nx=2,n_head=8,
                sig_max=1,mu_min=-3.0,mu_max=+3.0,dropout=0.3):
        self.name = name
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.k = k
        self.h_dim = h_dim
        self.n_head = n_head
        self.nx = nx
        self.sig_max = sig_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.dropout_rate = dropout
        self.build_model()

    def build_model(self):
        self.encoder = Encoder(self.x_dim, dk=int(self.h_dim/self.n_head), dv=int(self.h_dim/self.n_head)
                                ,d_model=self.h_dim,n_heads=self.n_head,dropout=self.dropout_rate, nx=self.nx)
        mog = MixturesOfGaussianLayer(self.h_dim,self.y_dim,self.k,sig_max=self.sig_max)
    
    def forward(self,x):
        z = self.encoder(x,None)
        z = torch.mean(z,axis=1) # [N x L x D] -> [N x D]
        z = self.mog(z)
        return z