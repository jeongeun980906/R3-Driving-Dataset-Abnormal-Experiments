from MDN.mha.common import MultiHeadAttentionLayer,Point_Wise_FeedForward,PositionalEncoding

import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.optim as optim
import math

class EncoderLayer(nn.Module):
    def __init__(self,dk=64, dv=64,d_model=512,n_heads=8,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.mha = MultiHeadAttentionLayer(dk=dk, dv=dv, d_model=d_model,n_heads=n_heads, dropout=dropout)
        self.pwf = Point_Wise_FeedForward(d_model = d_model,dropout=dropout)
    def forward(self,x,mask):
        query,key,value = x,x,x
        x_1,_ = self.mha(query,key,value,mask)
        x = self.LayerNorm(x+x_1)
        x_1 = self.pwf(x)
        x = self.LayerNorm(x+x_1)
        return x

class Encoder(nn.Module):
    def __init__(self,ntoken,dk=64, dv=64,d_model=512,n_heads=8,dropout=0.1, nx=2):
        super(Encoder,self).__init__()
        self.positional_encodding = PositionalEncoding(emsize=d_model,dropout=dropout)
        self.layers = nn.ModuleList([])
        self.embedding = nn.Embedding(ntoken, d_model)
        for _ in range(nx):
            self.layers.append(
                EncoderLayer(dk=dk, dv=dv,d_model=d_model,n_heads=n_heads,dropout=dropout)
            )
        self.layers = nn.Sequential(*self.layers)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to('cuda')
    
    def forward(self,x,mask=None):
        x = self.embedding(x)*self.scale
        x = self.positional_encodding(x)
        for layer in self.layers:
            x = layer(x,mask)
        return x
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # Heuristic: fc_mu.bias ~ Uniform(mu_min,mu_max)