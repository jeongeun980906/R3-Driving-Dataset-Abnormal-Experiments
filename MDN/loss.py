import math
import torch
from torch.autograd import Variable
import torch.distributions as TD

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
device = 'cuda'

def mdn_loss(pi,mu,sigma,data):
    """
    pi: [N x K]
    mu: [N x K x D]
    sigma: [N x K x D]
    data: [N x D]
    """
    data_usq = torch.unsqueeze(data,1) # [N x 1 x D]
    data_exp = data_usq.expand_as(sigma) # [N x K x D]
    probs = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data_exp-mu)/sigma)**2) / sigma # [N x K x D]
    probs_prod = torch.prod(probs,2) # [N x K]
    prob = torch.sum(probs_prod*pi,dim=1) # [N]
    prob = torch.clamp(prob,min=1e-8) # Clamp if the prob is to small
    nll = -torch.log(prob) # [N] 
    out = {'data_usq':data_usq,'data_exp':data_exp,
           'probs':probs,'probs_prod':probs_prod,'prob':prob,'nll':nll}
    return out

def mdn_sample(pi,mu,sigma):
    """
    pi: [N x K]
    mu: [N x K x D]
    sigma: [N x K x D]
    """
    categorical = TD.Categorical(pi)
    mixture_list = list(categorical.sample().data)
    _N,_D = sigma.size(0),sigma.size(2)
    eps = Variable(torch.empty(_N,_D).normal_()).to(device) # [N x D]
    sample = torch.empty_like(eps) # [N x D]
    for i_idx, mixture_idx in enumerate(mixture_list):
        mu_i,sigma_i = mu[i_idx,mixture_idx],sigma[i_idx,mixture_idx]
        sample[i_idx] = eps[i_idx].mul(sigma_i).add(mu_i)
    return sample # [N x D]

def mdn_uncertainties(pi, mu, sigma):
    # Compute Epistemic Uncertainty
    M = 0.1
    pi = torch.softmax(M*pi,1) # (optional) heuristics 
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D]
    mu_avg = torch.sum(torch.mul(pi_exp,mu),dim=1).unsqueeze(1) # [N x 1 x D]
    mu_exp = mu_avg.expand_as(mu) # [N x K x D]
    mu_diff_sq = torch.square(mu-mu_exp) # [N x K x D]
    epis_unct = torch.sum(torch.mul(pi_exp,mu_diff_sq), dim=1)  # [N x D]

    # Compute Aleatoric Uncertainty
    alea_unct = torch.sum(torch.mul(pi_exp,sigma), dim=1)  # [N x D]
    # Sqrt
    epis_unct,alea_unct = torch.sqrt(epis_unct),torch.sqrt(alea_unct)
    # entropy of pi
    entropy_pi  = -pi*torch.log(pi+1e-8)
    entropy_pi  = torch.sum(entropy_pi,1) #[N]
    out = {'epis':epis_unct,'alea':alea_unct,'pi_entropy':entropy_pi}
    return out
