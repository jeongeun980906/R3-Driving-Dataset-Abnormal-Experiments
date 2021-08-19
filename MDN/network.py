import torch.nn as nn
from collections import OrderedDict
import torch

class LinActv(nn.Module):
    def __init__(self,in_dim,out_dim,actv=nn.Tanh()):
        super(LinActv,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.actv = actv
        self.fc = nn.Sequential(
            OrderedDict([
                 ('lin',nn.Linear(self.in_dim,self.out_dim)),
                 ('actv',self.actv)
                ])
            )
    def forward(self,x):
        return self.fc(x)

class MixturesOfGaussianLayer(nn.Module):
    def __init__(self,in_dim,y_dim,k,sig_max=None):
        super(MixturesOfGaussianLayer,self).__init__()
        self.in_dim = in_dim
        self.y_dim = y_dim
        self.k = k
        self.sig_max = sig_max

        self.fc_pi = nn.Linear(self.in_dim,self.k)
        self.fc_mu = nn.Linear(self.in_dim,self.k*self.y_dim)
        self.fc_sigma = nn.Linear(self.in_dim,self.k*self.y_dim)

    def forward(self,x):
        pi_logit = self.fc_pi(x) # [N x K]
        pi = torch.softmax(pi_logit,1) # [N x K]
        mu = self.fc_mu(x) # [N x KD]
        mu = torch.reshape(mu,(-1,self.k,self.y_dim)) # [N x K x D]
        sigma = self.fc_sigma(x) # [N x KD]
        sigma = torch.reshape(sigma,(-1,self.k,self.y_dim)) # [N x K x D]
        if self.sig_max is None:
            sigma = torch.exp(sigma) # [N x K x D]
        else:
            sigma = self.sig_max * (torch.sigmoid(sigma) + 1e-8) # [N x K x D]
        out = {'pi':pi,'mu':mu,'sigma':sigma}
        return out

class MixtureDensityNetwork(nn.Module):
    def __init__(self,name='mdn',x_dim=1,y_dim=2,k=5,h_dims=[32,32],actv=nn.Tanh(),sig_max=1,
                 mu_min=-3.0,mu_max=+3.0,dropout=0.3):
        super(MixtureDensityNetwork,self).__init__()
        self.name = name
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.k = k
        self.h_dims = h_dims
        self.actv = actv
        self.sig_max = sig_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.dropout_rate = dropout
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        in_dim = self.x_dim
        for h_dim in self.h_dims:
            hidden_layer = LinActv(in_dim=in_dim,out_dim=h_dim,actv=self.actv)
            self.layers.append(hidden_layer)
            dropout = nn.Dropout(p=self.dropout_rate)
            self.layers.append(dropout)
            in_dim = h_dim
        # Final GMM
        mog = MixturesOfGaussianLayer(in_dim,self.y_dim,self.k,sig_max=self.sig_max)
        self.layers.append(mog)

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        """
        Heuristic: fc_mu.bias ~ Uniform(mu_min,mu_max)
        """
        self.layers[-1].fc_mu.bias.data.uniform_(self.mu_min,self.mu_max)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # [D,3,N,k]