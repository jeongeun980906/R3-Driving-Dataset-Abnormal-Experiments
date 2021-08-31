from MDN.network import LinActv,MixturesOfGaussianLayer
from MDN.mha.encoder import Encoder
import torch.nn as nn
from collections import OrderedDict
import torch

class MixtureDensityNetwork_MHA(nn.Module):
    def __init__(self,name='mdn',x_dim=1,y_dim=2,k=5,nx=2,n_head=8,
                sig_max=1,mu_min=-3.0,mu_max=+3.0,dropout=0.3):
        super(MixtureDensityNetwork_MHA, self).__init__()
        self.name = name
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.k = k
        self.n_head = n_head
        self.nx = nx
        self.sig_max = sig_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.dropout_rate = dropout
        self.build_model()

    def get_mask(self,src):
        mask = torch.ones_like(src,dtype=torch.int64)
        mask = torch.where(src<-1e6,mask,0)
        src = mask*src
        return mask,src

    def build_model(self):
        self.encoder = Encoder(self.x_dim, dk=self.x_dim, dv=self.x_dim
                                ,d_model=self.x_dim*self.n_head,n_heads=self.n_head,dropout=self.dropout_rate, nx=self.nx)
        self.mog = MixturesOfGaussianLayer(self.x_dim*self.n_head,self.y_dim,self.k,sig_max=self.sig_max)
    
    def forward(self,x):
        mask,x = self.get_mask(x) 
        z = self.encoder(x,mask)
        z = torch.mean(z,axis=1) # [N x L x D] -> [N x D]
        z = self.mog(z)
        return z

    def init_param(self):
        self.encoder.init_param()
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        """
        Heuristic: fc_mu.bias ~ Uniform(mu_min,mu_max)
        """
        self.mog.fc_mu.bias.data.uniform_(self.mu_min,self.mu_max)