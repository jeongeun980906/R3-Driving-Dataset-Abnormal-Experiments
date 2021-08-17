import torch
import torch.optim as optim
import torch.nn as nn
import random
from tools.utils import print_n_txt

from MDN.loss import mdn_loss,mdn_sample,mdn_uncertainties
from MDN.network import MixtureDensityNetwork
from MDN.eval import query_mdn,test_eval_mdn

class solver():
    def __init__(self,args,device):
        self.EPOCH = args.epoch
        self.device = device
        self.load_model(args)
        self.method = args.query_method
        self.query_size = args.query_size
        self.query_init_weight = args.init_weight
        self.lr = args.lr
        self.wd = args.wd
        self.lr_rate = args.lr_rate
        self.lr_step = args.lr_step

    def load_model(self,args):
        self.model = MixtureDensityNetwork(
                    name='mdn',x_dim=32, y_dim=3,k=args.k,h_dims=[64,64],actv=nn.ReLU(),sig_max=args.sig_max,
                    mu_min=-1,mu_max=+1,dropout=args.dropout).to(self.device)

    def init_param(self):
        self.model.init_param()

    def train_mdn(self,train_iter,test_exp_iter,test_neg_iter,f):
        if self.query_init_weight:
            self.init_param()
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.wd,eps=1e-8)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=args.lr_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.lr_rate, step_size=self.lr_step)
        train_nll = []
        test_nll = []
        for epoch in range(self.EPOCH):
            loss_sum = 0.0
            #time.sleep(1)
            for batch_in,batch_out in train_iter:
                out =  self.model.forward(batch_in.to(self.device))
                pi,mu,sigma = out['pi'],out['mu'],out['sigma']
                loss_out = mdn_loss(pi,mu,sigma,batch_out.to(self.device)) # 'mace_avg','epis_avg','alea_avg'
                loss = torch.mean(loss_out['nll'])
                optimizer.zero_grad() # reset gradient
                loss.backward() # back-propagation
                optimizer.step() # optimizer update
                # Track losses
                loss_sum += loss
            scheduler.step()
            loss_avg = loss_sum/len(train_iter)
            train_out = test_eval_mdn(self.model,train_iter,'cuda')
            test_in_out = test_eval_mdn(self.model,test_exp_iter,'cuda')
            test_ood_out = test_eval_mdn(self.model,test_neg_iter,'cuda')

            strTemp = ("epoch: [%d/%d] loss: [%.3f] train_nll:[%.4f] test_nll: [%.4f]"
                        %(epoch,self.EPOCH,loss_avg,train_out['nll'],test_in_out['nll']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[ID] epis avg: [%.3f] alea avg: [%.3f] pi_entropy avg: [%.3f]"%
                (test_in_out['epis'],test_in_out['alea'],test_in_out['pi_entropy']))
            print_n_txt(_f=f,_chars=strTemp)

            strTemp =  ("[OOD] epis avg: [%.3f] alea avg: [%.3f] pi_entropy avg: [%.3f]"%
                    (test_ood_out['epis'],test_ood_out['alea'],test_ood_out['pi_entropy']))
            print_n_txt(_f=f,_chars=strTemp)
            train_nll.append(train_out['nll'])
            test_nll.append(test_in_out['nll'])
        if epoch>5:
            train_nll = train_nll[-5:]
            test_nll = test_nll[-5:]
        train_nll = sum(train_nll)/len(train_nll)
        test_nll = sum(test_nll)/len(test_nll)
        return train_nll,test_nll

    def query_data(self,unlabel_iter,unl_size=None):
        out = query_mdn(self.model,unlabel_iter,'cuda')
        if self.method == 'epistemic':
            out = out['epis_']
        elif self.method == 'aleatoric':
            out = out['alea_']
        elif self.method == 'pi_entropy':
            out = out['pi_entropy_']
        elif self.method == 'random':
            return  torch.tensor(random.sample(range(unl_size), self.query_size))
        else:
            raise NotImplementedError()
        out  = torch.FloatTensor(out)
        _, max_idx = torch.topk(out,self.query_size,0)
        max_idx = max_idx.type(torch.LongTensor)
        return max_idx