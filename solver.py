import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import print_n_txt
from VAE.variants import WAE, RAE, VQVAE
from MDN.loss import mdn_loss,mdn_eval,mdn_uncertainties
from MDN.network import MixtureDensityNetwork
from VAE.network import VAE
from VAE.loss import VAE_loss,VAE_eval
from tools.dataloader import mixquality_dataset

import time
class solver():
    def __init__(self,args,device,SEED):
        self.EPOCH = args.epoch
        self.device = device
        self.lr = args.lr
        self.wd = args.wd
        self.lr_rate = args.lr_rate
        self.lr_step = args.lr_step
        self.CLIP = 1
        self.SEED=SEED
        self.load_iter(args)
        self.load_model(args)

    def load_model(self,args):
        if args.mode== 'vae':
            self.model = VAE(x_dim=self.data_dim[0],h_dim=args.h_dim,z_dim=args.z_dim).to(self.device)
            self.train_func = self.train_VAE
            self.eval_func = self.eval_ood_VAE
        elif args.mode == 'mdn':
            self.model = MixtureDensityNetwork(
                    name='mdn',x_dim=self.data_dim[0], y_dim=self.data_dim[1],k=args.k,h_dims=[128,128],actv=nn.ReLU(),sig_max=args.sig_max,
                    mu_min=-3,mu_max=+3,dropout=args.dropout).to(self.device)
            self.train_func = self.train_mdn
            self.eval_func = self.eval_ood_mdn
        # === NEW: WAE ===
        elif args.mode == 'wae':
            self.model      = WAE(x_dim=self.data_dim[0],
                                   h_dim=args.h_dim,
                                   z_dim=args.z_dim,
                                   lambda_mmd=args.lambda_mmd,
                                   sigma=args.sigma).to(self.device)
            self.train_func = self.train_WAE
            self.eval_func  = self.eval_ood_WAE

        # === NEW: RAE ===
        elif args.mode == 'rae':
            self.model      = RAE(x_dim=self.data_dim[0],
                                   h_dim=args.h_dim,
                                   z_dim=args.z_dim,
                                   lambda_z=args.lambda_z).to(self.device)
            self.train_func = self.train_RAE
            self.eval_func  = self.eval_ood_RAE

        # === NEW: VQ-VAE ===
        elif args.mode == 'vqvae':
            self.model      = VQVAE(x_dim=self.data_dim[0],
                                     h_dim=args.h_dim,
                                     z_dim=args.z_dim,
                                     num_embeddings=args.num_embeddings,
                                     commitment_cost=args.commitment_cost).to(self.device)
            self.train_func = self.train_VQVAE
            self.eval_func  = self.eval_ood_VQVAE


    def init_param(self):
        self.model.init_param()
    
    def load_iter(self,args):
        root = args.root+'/R3-Driving-Dataset/dataset/'
        print("Loading Dataset")
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
        print("Done!")

    def train_VAE(self, f):
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.wd,eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)
        train_l2 = []
        test_l2 = []
        for epoch in range(self.EPOCH):
            loss_sum = 0.0
            for batch_in,batch_out in self.train_iter:
                #batch_in = torch.cat((batch_in,batch_out),dim=1)
                x_reconst, mu, logvar =  self.model.forward(batch_in.to(self.device))
                loss_out = VAE_loss(batch_in.to(self.device), x_reconst, mu, logvar)
                loss = torch.mean(loss_out['loss'])
                optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP)
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
            for batch_in,batch_out in data_iter:
                #batch_in = torch.cat((batch_in,batch_out),dim=1)
                x_recon, mu, logvar = self.model.forward(batch_in.to(device))
                loss_out = VAE_eval(batch_in.to(self.device), x_recon, mu, logvar)
                recon   = loss_out['reconst_loss'] # [N x D]
                kl  = loss_out['kl_div'] # [N x D]
                recon_ += recon.cpu().numpy().tolist()
                kl_ += kl.cpu().numpy().tolist()
                n_total += batch_in.size(0)
            self.model.train() # back to train mode
            out_eval = {'recon_' : recon_,'kl_' : kl_}
        return out_eval

    def test_eval_VAE (self, data_iter, device):
        with torch.no_grad():
            n_total,recon,kl_div,total_loss = 0,0,0,0
            self.model.eval() 
            for batch_in,batch_out in data_iter:
                #batch_in = torch.cat((batch_in,batch_out),dim=1)
                x_reconst, mu, logvar = self.model.forward(batch_in.to(device))
                loss_out = VAE_loss(batch_in.to(self.device), x_reconst, mu, logvar)
                recon += torch.sum(loss_out['reconst_loss'])
                kl_div += torch.sum(loss_out['kl_div'])
                total_loss += torch.sum(loss_out['loss'])
                n_total += batch_in.size(0)
            recon_avg=(recon/n_total).detach().cpu().item()
            kl_avg = (kl_div/n_total).detach().cpu().item()
            total_avg = (total_loss/n_total).detach().cpu().item()
            self.model.train() # back to train mode
            out_eval = {'recon':recon_avg,'kl_div':kl_avg, 'total':total_avg}
        return out_eval

    # ----- WAE -----
    def train_WAE(self, f):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)
        train_losses, id_losses = [], []

        for epoch in range(self.EPOCH):
            epoch_loss = 0.0
            for x, _ in self.train_iter:
                x = x.to(self.device)
                # forward
                x_recon, mu, logvar = self.model(x)
                # compute WAE loss (uses compute_mmd under the hood)
                loss, info = self.model.loss_function(x_recon, x, mu)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            # evaluate on ID / OOD
            id_stats  = self.test_eval_WAE(self.test_e_iter, self.device)
            ood_stats = self.test_eval_WAE(self.test_n_iter, self.device)

            msg = (f"[WAE][Epoch {epoch+1}/{self.EPOCH}]  "
                   f"Loss={epoch_loss/len(self.train_iter):.4f}  "
                   f"ID recon={id_stats['recon']:.4f}, MMD={id_stats['mmd']:.4f}  "
                   f"OOD recon={ood_stats['recon']:.4f}, MMD={ood_stats['mmd']:.4f}")
            print_n_txt(f, msg)

            train_losses.append(epoch_loss/len(self.train_iter))
            id_losses.append(id_stats['loss'])

        # (optional) save final weights
        torch.save(self.model.state_dict(), f"wae_{self.SEED}.pth")
        return train_losses, id_losses

    def test_eval_WAE(self, data_iter, device):
        with torch.no_grad():
            recon_sum, mmd_sum, total_sum, n = 0.0, 0.0, 0.0, 0
            self.model.eval()
            for x, _ in data_iter:
                x = x.to(device)
                x_recon, mu, _ = self.model(x)
                loss, info     = self.model.loss_function(x_recon, x, mu)
                batch_size     = x.size(0)
                recon_sum += info['recon'] * batch_size
                mmd_sum   += info['mmd']   * batch_size
                total_sum += loss.item()   * batch_size
                n += batch_size
            self.model.train()
            return {
                'recon': recon_sum / n,
                'mmd':   mmd_sum   / n,
                'loss':  total_sum / n
            }

    # ----- RAE -----
    def train_RAE(self, f):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)
        train_losses, id_losses = [], []

        for epoch in range(self.EPOCH):
            epoch_loss = 0.0
            for x, _ in self.train_iter:
                x = x.to(self.device)
                x_recon, mu, logvar = self.model(x)
                loss, info = self.model.loss_function(x_recon, x, mu)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            id_stats  = self.test_eval_RAE(self.test_e_iter, self.device)
            ood_stats = self.test_eval_RAE(self.test_n_iter, self.device)

            msg = (f"[RAE][Epoch {epoch+1}/{self.EPOCH}]  "
                   f"Loss={epoch_loss/len(self.train_iter):.4f}  "
                   f"ID recon={id_stats['recon']:.4f}, ZReg={id_stats['zreg']:.4f}  "
                   f"OOD recon={ood_stats['recon']:.4f}, ZReg={ood_stats['zreg']:.4f}")
            print_n_txt(f, msg)

            train_losses.append(epoch_loss/len(self.train_iter))
            id_losses.append(id_stats['loss'])

        torch.save(self.model.state_dict(), f"rae_{self.SEED}.pth")
        return train_losses, id_losses

    def test_eval_RAE(self, data_iter, device):
        with torch.no_grad():
            recon_sum, zreg_sum, total_sum, n = 0.0, 0.0, 0.0, 0
            self.model.eval()
            for x, _ in data_iter:
                x = x.to(device)
                x_recon, mu, _ = self.model(x)
                loss, info     = self.model.loss_function(x_recon, x, mu)
                batch_size     = x.size(0)
                recon_sum += info['recon'] * batch_size
                zreg_sum  += info['zreg'] * batch_size
                total_sum += loss.item()   * batch_size
                n += batch_size
            self.model.train()
            return {
                'recon': recon_sum / n,
                'zreg':  zreg_sum  / n,
                'loss':  total_sum / n
            }

    # ----- VQ-VAE -----
    def train_VQVAE(self, f):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)
        train_losses, id_losses = [], []

        for epoch in range(self.EPOCH):
            epoch_loss = 0.0
            for x, _ in self.train_iter:
                x = x.to(self.device)
                x_recon, z_e, z_q, vq_loss = self.model(x)
                loss, info = self.model.loss_function(x_recon, x, z_e, z_q, vq_loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP)
                optimizer.step()
                epoch_loss += -loss.item()
            scheduler.step()

            id_stats  = self.test_eval_VQVAE(self.test_e_iter, self.device)
            ood_stats = self.test_eval_VQVAE(self.test_n_iter, self.device)

            msg = (f"[VQ-VAE][Epoch {epoch+1}/{self.EPOCH}]  "
                   f"Loss={epoch_loss/len(self.train_iter):.4f}  "
                   f"ID recon={id_stats['recon']:.4f}, VQLoss={id_stats['vq_loss']:.4f}  "
                   f"OOD recon={ood_stats['recon']:.4f}, VQLoss={ood_stats['vq_loss']:.4f}")
            print_n_txt(f, msg)

            train_losses.append(epoch_loss/len(self.train_iter))
            id_losses.append(id_stats['loss'])

        torch.save(self.model.state_dict(), f"vqvae_{self.SEED}.pth")
        return train_losses, id_losses

    def test_eval_VQVAE(self, data_iter, device):
        with torch.no_grad():
            recon_sum, vq_sum, total_sum, n = 0.0, 0.0, 0.0, 0
            self.model.eval()
            for x, _ in data_iter:
                x = x.to(device)
                x_recon, z_e, z_q, vq_loss = self.model(x)
                loss, info                  = self.model.loss_function(x_recon, x, z_e, z_q, vq_loss)
                batch_size                  = x.size(0)
                recon_sum   += info['recon']   * batch_size
                vq_sum      += info['vq_loss'] * batch_size
                total_sum   += loss.item()     * batch_size
                n += batch_size
            self.model.train()
            return {
                'recon':   recon_sum   / n,
                'vq_loss': vq_sum      / n,
                'loss':    total_sum   / n
            }

    def eval_ood_WAE(self, data_iter, device):
        import torch.nn.functional as F
        from VAE.variants import compute_mmd

        recon_list, mmd_list = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_in, _ in data_iter:
                x = batch_in.to(device)
                x_recon, mu, _ = self.model(x)

                # per-sample reconstruction error
                recon_err = F.mse_loss(x_recon, x, reduction='none')
                recon_err = recon_err.view(x.size(0), -1).sum(dim=1)

                # batch-level MMD → replicate per sample
                prior = torch.randn_like(mu)
                mmd_val = compute_mmd(mu, prior, sigma=self.model.sigma)

                recon_list += recon_err.cpu().tolist()
                mmd_list   += [mmd_val.item()] * x.size(0)

        self.model.train()
        return {'recon_': recon_list, 'mmd_': mmd_list}


    def eval_ood_RAE(self, data_iter, device):
        import torch.nn.functional as F

        recon_list, zreg_list = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_in, _ in data_iter:
                x = batch_in.to(device)
                x_recon, mu, _ = self.model(x)

                # per-sample reconstruction error
                recon_err = F.mse_loss(x_recon, x, reduction='none')
                recon_err = recon_err.view(x.size(0), -1).sum(dim=1)

                # per-sample z-regularizer
                zreg = mu.pow(2).sum(dim=1)

                recon_list += recon_err.cpu().tolist()
                zreg_list  += zreg.cpu().tolist()

        self.model.train()
        return {'recon_': recon_list, 'zreg_': zreg_list}


    def eval_ood_VQVAE(self, data_iter, device):
        import torch.nn.functional as F

        recon_list, vq_list = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_in, _ in data_iter:
                x = batch_in.to(device)
                x_recon, z_e, z_q, vq_loss = self.model(x)

                # per-sample reconstruction error
                recon_err = F.mse_loss(x_recon, x, reduction='none')
                recon_err = recon_err.view(x.size(0), -1).sum(dim=1)

                # batch-level VQ loss → replicate per sample
                recon_list += recon_err.cpu().tolist()
                vq_list    += [vq_loss.item()] * x.size(0)

        self.model.train()
        return {'recon_': recon_list, 'vq_': vq_list}


    def train_mdn(self,f):
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.wd,eps=1e-8)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)
        train_l2 = []
        test_l2 = []
        for epoch in range(self.EPOCH):
            loss_sum = 0.0
            #time.sleep(1)
            for batch_in,batch_out in self.train_iter:
                out =  self.model.forward(batch_in.to(self.device))
                pi,mu,sigma = out['pi'],out['mu'],out['sigma']
                loss_out = mdn_loss(pi,mu,sigma,batch_out.to(self.device)) # 'mace_avg','epis_avg','alea_avg'
                loss = torch.mean(loss_out['nll'])
                optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP)
                optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss
            scheduler.step()
            loss_avg = loss_sum/len(self.train_iter)
            train_out = self.test_eval_mdn(self.train_iter,'cuda')
            test_in_out = self.test_eval_mdn(self.test_e_iter,'cuda')
            test_ood_out = self.test_eval_mdn(self.test_n_iter,'cuda')

            strTemp = ("epoch: [%d/%d] loss: [%.3f] train_l2:[%.4f] test_l2: [%.4f]"
                        %(epoch,self.EPOCH,loss_avg,train_out['l2_norm'],test_in_out['l2_norm']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[ID] epis avg: [%.3f] alea avg: [%.3f] pi_entropy avg: [%.3f]"%
                (test_in_out['epis'],test_in_out['alea'],test_in_out['pi_entropy']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[OOD] epis avg: [%.3f] alea avg: [%.3f] pi_entropy avg: [%.3f]"%
                    (test_ood_out['epis'],test_ood_out['alea'],test_ood_out['pi_entropy']))
            print_n_txt(_f=f,_chars=strTemp)
            train_l2.append(train_out['l2_norm'])
            test_l2.append(test_in_out['l2_norm'])
        return train_l2,test_l2

    def eval_ood_mdn(self,data_iter,device):
        with torch.no_grad():
            n_total= 0
            pi_entropy_ , epis_ ,alea_  = list(),list(),list()
            self.model.eval() # evaluate (affects DropOut and BN)
            for batch_in,_ in data_iter:
                # Foraward path
                mdn_out     = self.model.forward(batch_in.to(device))
                pi,mu,sigma = mdn_out['pi'],mdn_out['mu'],mdn_out['sigma']

                unct_out    = mdn_uncertainties(pi,mu,sigma)
                epis_unct   = unct_out['epis'] # [N x D]
                alea_unct   = unct_out['alea'] # [N x D]
                pi_entropy  = unct_out['pi_entropy'] # [N]

                # epis_unct = torch.mean(epis_unct,dim=-1)
                # alea_unct = torch.mean(alea_unct,dim=-1)
                
                epis_unct,_ = torch.max(epis_unct,dim=-1)
                alea_unct,_ = torch.max(alea_unct,dim=-1)        
                
                epis_ += epis_unct.cpu().numpy().tolist()
                alea_ += alea_unct.cpu().numpy().tolist()
                pi_entropy_ += pi_entropy.cpu().numpy().tolist()

                n_total += batch_in.size(0)
            self.model.train() # back to train mode 
            out_eval = {'epis_' : epis_,'alea_' : alea_,'pi_entropy_':pi_entropy_}
        return out_eval
    
    def test_eval_mdn(self,data_iter,device):
        with torch.no_grad():
            n_total,l2_sum,epis_unct_sum,alea_unct_sum,entropy_pi_sum = 0,0,0,0,0
            self.model.eval() # evaluate (affects DropOut and BN)
            for batch_in,batch_out in data_iter:
                # Foraward path
                mdn_out     = self.model.forward(batch_in.to(device))
                pi,mu,sigma = mdn_out['pi'],mdn_out['mu'],mdn_out['sigma']

                l2        = mdn_eval(pi,mu,sigma,batch_out.to(device))['l2_mean']
                unct_out    = mdn_uncertainties(pi,mu,sigma)
                epis_unct   = unct_out['epis'] # [N x D]
                alea_unct   = unct_out['alea'] # [N x D]
                entropy_pi  = unct_out['pi_entropy'] # [N]
                entropy_pi_sum  += torch.sum(entropy_pi)
                epis_unct,_ = torch.max(epis_unct,dim=-1)
                alea_unct,_ = torch.max(alea_unct,dim=-1)
                # epis_unct_sum += torch.sum(torch.mean(epis_unct,dim=-1))
                # alea_unct_sum += torch.sum(torch.mean(alea_unct,dim=-1))
                epis_unct_sum += torch.sum(epis_unct,dim=-1)
                alea_unct_sum += torch.sum(alea_unct,dim=-1)

                l2_sum += torch.sum(l2) # [N]
                n_total += batch_in.size(0)
            entropy_pi_avg=(entropy_pi_sum/n_total).detach().cpu().item()
            epis      = (epis_unct_sum/n_total).detach().cpu().item()
            alea      = (alea_unct_sum/n_total).detach().cpu().item()
            l2      = (l2_sum/n_total).detach().cpu().item()
            self.model.train() # back to train mode 
            out_eval = {'l2_norm':l2,'epis':epis,'alea':alea,
                        'pi_entropy':entropy_pi_avg}
        return out_eval