import torch
from MDN.loss import mdn_loss,mdn_uncertainties 
import matplotlib.pyplot as plt
import numpy as np

def test_eval_mdn(model,data_iter,device):
    with torch.no_grad():
        n_total,nll_sum,epis_unct_sum,alea_unct_sum,entropy_pi_sum = 0,0,0,0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            # Foraward path
            mdn_out     = model.forward(batch_in.to(device))
            pi,mu,sigma = mdn_out['pi'],mdn_out['mu'],mdn_out['sigma']

            nll        = mdn_loss(pi,mu,sigma,batch_out.to(device))['nll']
            unct_out    = mdn_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N x D]
            alea_unct   = unct_out['alea'] # [N x D]
            entropy_pi  = unct_out['pi_entropy'] # [N]
            entropy_pi_sum  += torch.sum(entropy_pi)
            epis_unct_sum += torch.sum(torch.mean(epis_unct),dim=-1)
            alea_unct_sum += torch.sum(torch.mean(alea_unct,dim=-1))
            nll_sum += torch.sum(nll) # [N]
            n_total += batch_in.size(0)
        entropy_pi_avg=(entropy_pi_sum/n_total).detach().cpu().item()
        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        nll      = (nll_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'nll':nll,'epis':epis,'alea':alea,
                    'pi_entropy':entropy_pi_avg}
    return out_eval

def query_mdn(model,data_iter,device):
    with torch.no_grad():
        n_total= 0
        pi_entropy_ , epis_ ,alea_  = list(),list(),list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in in data_iter:
            # Foraward path
            mdn_out     = model.forward(batch_in.to(device))
            pi,mu,sigma = mdn_out['pi'],mdn_out['mu'],mdn_out['sigma']

            unct_out    = mdn_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N x D]
            alea_unct   = unct_out['alea'] # [N x D]
            pi_entropy  = unct_out['pi_entropy'] # [N]

            epis_unct = torch.mean(epis_unct,dim=-1)
            alea_unct = torch.mean(alea_unct,dim=-1)
            
            epis_ += list(epis_unct.cpu().numpy())
            alea_ += list(alea_unct.cpu().numpy())
            pi_entropy_ += list(pi_entropy.cpu().numpy())

            n_total += batch_in.size(0)
        model.train() # back to train mode 
        out_eval = {'epis_' : epis_,'alea_' : alea_,'pi_entropy_':pi_entropy_}
    return out_eval

def eval_ood_mdn(model,data_iter,device):
    with torch.no_grad():
        n_total= 0
        pi_entropy_ , epis_ ,alea_  = list(),list(),list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,_ in data_iter:
            # Foraward path
            mdn_out     = model.forward(batch_in.to(device))
            pi,mu,sigma = mdn_out['pi'],mdn_out['mu'],mdn_out['sigma']

            unct_out    = mdn_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N x D]
            alea_unct   = unct_out['alea'] # [N x D]
            pi_entropy  = unct_out['pi_entropy'] # [N]

            epis_unct = torch.mean(epis_unct,dim=-1)
            alea_unct = torch.mean(alea_unct,dim=-1)
            
            epis_ += list(epis_unct.cpu().numpy())
            alea_ += list(alea_unct.cpu().numpy())
            pi_entropy_ += list(pi_entropy.cpu().numpy())

            n_total += batch_in.size(0)
        model.train() # back to train mode 
        out_eval = {'epis_' : epis_,'alea_' : alea_,'pi_entropy_':pi_entropy_}
    return out_eval