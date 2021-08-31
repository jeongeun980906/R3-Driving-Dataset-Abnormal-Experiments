import torch
import torch.optim as optim
import torch.nn as nn
from tools.utils import print_n_txt

from VAE.loss import VAE_loss
from VAE.network import VAE
from tools.dataloader import mixquality_dataset
class solver():
    def __init__(self,args,device,SEED):
        self.EPOCH = args.epoch
        self.device = device
        self.lr = args.lr
        self.wd = args.wd
        self.lr_rate = args.lr_rate
        self.lr_step = args.lr_step
        self.CLIP = 0.1
        self.SEED=SEED
        self.load_iter(args)
        self.load_model(args)

    def load_model(self,args):
        if args.transformer:
            self.model = MixtureDensityNetwork_MHA(
                    name='mdn',x_dim=self.data_dim[0], y_dim=self.data_dim[1],k=args.k,n_head=3,nx=2
                    ,sig_max=args.sig_max,mu_min=-3,mu_max=+3,dropout=args.dropout).to(self.device)
        else:
            self.model = VAE().to(self.device)

    def init_param(self):
        self.model.init_param()
    
    def load_iter(self,args):
        if args.transformer:
            if args.light:
                root = args.root+'/light_mixquality/'
            else:
                root = args.root+'/mixquality/'
            self.train_dataset = mixquality_dataset_mha(root = root, train=True,norm=args.norm,frame=args.frame,exp_case=args.exp_case)
            self.train_iter = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.batch_size, 
                                    shuffle=False)
            torch.manual_seed(self.SEED)
            self.test_e_dataset = mixquality_dataset_mha(root = root, train=False,neg=False,norm=args.norm,frame=args.frame,exp_case=args.exp_case)
            self.test_e_iter = torch.utils.data.DataLoader(self.test_e_dataset, batch_size=args.batch_size, 
                                    shuffle=False)
            torch.manual_seed(self.SEED)
            self.test_n_dataset = mixquality_dataset_mha(root = root, train=False,neg=True,norm=args.norm,frame=args.frame,exp_case=args.exp_case)
            self.test_n_iter = torch.utils.data.DataLoader(self.test_n_dataset, batch_size=args.batch_size, 
                                    shuffle=False)
        else:
            if args.light:
                root = args.root+'/light_mixquality/'
            else:
                root = args.root+'/mixquality/'
            self.train_dataset = mixquality_dataset(root = root, train=True,norm=args.norm,frame=args.frame,exp_case=args.exp_case)
            self.train_iter = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.batch_size, 
                                    shuffle=False)
            torch.manual_seed(self.SEED)
            self.test_e_dataset = mixquality_dataset(root = root, train=False,neg=False,norm=args.norm,frame=args.frame,exp_case=args.exp_case)
            self.test_e_iter = torch.utils.data.DataLoader(self.test_e_dataset, batch_size=args.batch_size, 
                                    shuffle=False)
            torch.manual_seed(self.SEED)
            self.test_n_dataset = mixquality_dataset(root = root, train=False,neg=True,norm=args.norm,frame=args.frame,exp_case=args.exp_case)
            self.test_n_iter = torch.utils.data.DataLoader(self.test_n_dataset, batch_size=args.batch_size, 
                                    shuffle=False)

        self.data_dim = [self.train_dataset.x.size(-1), self.train_dataset.y.size(-1)]

    def train_VAE(self, f):
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.wd,eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)
        train_l2 = []
        test_l2 = []
        for epoch in range(self.EPOCH):
            loss_sum = 0.0
            for batch_in,batch_out in self.train_iter:
                x_reconst, mu, logvar =  self.model.forward(batch_in.to(self.device))
                loss_out = VAE_loss(batch_in.to(self.device), x_reconst, mu, logvar)
                loss = torch.mean(loss_out['loss'])
                optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP)
                optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss
            scheduler.step()
            loss_avg = loss_sum/len(self.train_iter)
            train_out = self.test_eval_VAE(self.train_iter,'cuda')
            test_in_out = self.test_eval_VAE(self.test_e_iter,'cuda')
            test_ood_out = self.test_eval_VAE(self.test_n_iter,'cuda')
            strTemp = ("epoch: [%d/%d] loss: [%.3f] train_loss:[%.4f] test_loss: [%.4f]"
                        %(epoch,self.EPOCH,loss_avg,train_out['total'],test_in_out['total']))
            print_n_txt(_f=f,_chars=strTemp)
            strTemp =  ("[ID] recon avg: [%.3f] kl_div avg: [%.3f]"%
                (test_in_out['recon'],test_in_out['kl_div']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[OOD] recon avg: [%.3f] kl_div avg: [%.3f]"%
                    (test_ood_out['recon'],test_ood_out['kl_div']))
            print_n_txt(_f=f,_chars=strTemp)
            train_l2.append(train_out['total'])
            test_l2.append(test_in_out['total'])
        return train_l2,test_l2

    def eval_ood_VAE(self,data_iter,device):
        with torch.no_grad():
            n_total= 0
            recon_ , kl_  = list(),list()
            self.model.eval() # evaluate (affects DropOut and BN)
            for batch_in,_ in data_iter:
                x_recon, mu, logvar = self.model.forward(batch_in.to(device))
                loss_out = VAE_loss(batch_in.to(self.device), x_recon, mu, logvar)
                recon   = loss_out['reconst_loss'] # [N x D]
                kl  = loss_out['kl_div'] # [N x D]
                recon_ += recon.cpu().numpy().reshape(-1).tolist()
                kl_ += kl.cpu().numpy().reshape(-1).tolist()
                n_total += batch_in.size(0)
            self.model.train() # back to train mode
            out_eval = {'recon_' : recon_,'kl_' : kl_}
        return out_eval

    def test_eval_VAE (self, data_iter, device):
        with torch.no_grad():
            n_total,recon,kl_div,total_loss = 0,0,0,0
            self.model.eval() 
            for batch_in,batch_out in data_iter:
                x_reconst, mu, logvar = self.model.forward(batch_in.to(device))
                loss_out = VAE_loss(batch_in.to(self.device), x_reconst, mu, logvar)
                recon += loss_out['reconst_loss']
                kl_div += loss_out['kl_div']
                total_loss += loss_out['loss']
                n_total += batch_in.size(0)
            recon_avg=(recon/n_total).detach().cpu().item()
            kl_avg = (kl_div/n_total).detach().cpu().item()
            total_avg = (total_loss/n_total).detach().cpu().item()
            self.model.train() # back to train mode
            out_eval = {'recon':recon_avg,'kl_div':kl_avg, 'total':total_avg}
        return out_eval
