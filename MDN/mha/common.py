import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):

    def __init__(self, emsize, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emsize)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emsize, 2).float() * (-math.log(10000.0) / emsize))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,dk=64, dv=64,d_model=512,n_heads=8, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.fc_q = nn.Linear(dk*n_heads,d_model)
        self.fc_k = nn.Linear(dk*n_heads,d_model)
        self.fc_v = nn.Linear(dv*n_heads,d_model)
        self.scale = dk**(0.5)
        self.fc_l = nn.Linear(dv*n_heads,d_model)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.dk = dk
        self.dv = dv
        self.dmodel = d_model

    def forward(self,query,key,value,mask=None):
        batchsize = query.size(0)
        #print(query.size(1),key.size(1),value.size(1))
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batchsize,-1,self.n_heads,self.dk).permute(0,2,1,3)  # [N x H x L x dk]
        K = V.view(batchsize,-1,self.n_heads,self.dk).permute(0,2,1,3) # [N x H x L x dk]
        V = V.view(batchsize,-1,self.n_heads,self.dv).permute(0,2,1,3) # [N x H x L x dv]

        x = torch.matmul(Q,K.permute(0,1,3,2))/self.scale # [N x H x L x L]
        if mask is not None:
            try:
                x = x.masked_fill(mask==0,-1e9)
            except:
                pass
        attention = torch.softmax(x,dim=-1)
        x = torch.matmul(attention,V) # [N x H x L x dv]
        x = self.dropout(x)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batchsize,-1,self.dv*self.n_heads) # concat head
        x= self.fc_l(x) # [N x L x dmodel]
         
        return x,attention

class Point_Wise_FeedForward(nn.Module):
    def __init__(self,d_model=512,hdim=512,dropout=0.1):
        super(Point_Wise_FeedForward,self).__init__()
        self.fc1 = nn.Linear(d_model,hdim)
        self.fc2 = nn.Linear(hdim,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x