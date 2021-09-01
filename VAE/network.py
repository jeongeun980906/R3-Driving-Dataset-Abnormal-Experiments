import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, x_dim=173, h_dim=[20], z_dim=10):
        super(VAE, self).__init__()
        self.encoder_layer = nn.ModuleList()
        print(h_dim)
        previous = x_dim
        for h in h_dim:
            self.encoder_layer.append(
                nn.Linear(previous,h))
            self.encoder_layer.append(
                nn.ReLU()
            )
            previous = h
        self.fc2 = nn.Linear(h_dim[-1], z_dim)
        self.fc3 = nn.Linear(h_dim[-1], z_dim)
        self.decoder_layer = nn.ModuleList()
        previous = z_dim
        for h in reversed(h_dim):
            self.decoder_layer.append(
                nn.Linear(previous,h)
            )
            self.decoder_layer.append(nn.ReLU())
            previous = h
        self.fc5 = nn.Linear(h_dim[0], x_dim)

    def encode(self, x):
        for layer in self.encoder_layer:
            x = layer(x)
        return self.fc2(x), self.fc3(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for layer in self.decoder_layer:
            z = layer(z)
        return self.fc5(z) #F.sigmoid(self.fc5(z))

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
