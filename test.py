from tools.pool import AL_pool
from solver import solver
from tools.dataloader import total_dataset
from tools.utils import print_n_txt,Logger,filter_expert
from tools.measure_ood import measure,get_method
import torch
import torch.nn as nn
import random
import numpy as np

import argparse
import os
from MDN.eval import eval_ood_mdn
from MDN.network import MixtureDensityNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,default='./dataset',help='root directory of the dataset')
parser.add_argument('--id', type=int,default=1,help='id')
parser.add_argument('--gpu', type=int,default=0,help='gpu id')

parser.add_argument('--query_step', type=int,default=10,help='query step')
parser.add_argument('--query_size', type=int,default=1000,help='query size')
parser.add_argument('--init_pool', type=int,default=2000,help='number of initial data')
parser.add_argument('--query_method', type=str,default='epistemic',help='query method')
parser.add_argument('--epoch', type=int,default=100,help='epoch')
parser.add_argument('--init_weight', type=bool,default=True,help='init weight on every query step')

parser.add_argument('--lr', type=float,default=1e-3,help='learning rate')
parser.add_argument('--batch_size', type=int,default=128,help='batch size')
parser.add_argument('--wd', type=float,default=1e-4,help='weight decay')

parser.add_argument('--k', type=int,default=10,help='number of mixtures')
parser.add_argument('--sig_max', type=float,default=1,help='sig max')

args = parser.parse_args()

SEED = 0
EPOCH = args.epoch

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device='cuda'

p = AL_pool(root=args.root,num_init=args.init_pool)
test_e_dataset = total_dataset(root = args.root, train=False,neg=False)
test_e_iter = torch.utils.data.DataLoader(test_e_dataset, batch_size=args.batch_size, 
                        shuffle=False)
test_n_dataset = total_dataset(root = args.root, train=False,neg=True)
test_n_iter = torch.utils.data.DataLoader(test_n_dataset, batch_size=args.batch_size, 
                        shuffle=False)

DIR = './res/mdn_{}/{}/ckpt/'.format(args.query_method,args.id)
state_dict = torch.load(DIR+'9.pth')
model = MixtureDensityNetwork(
                    name='mdn',x_dim=32, y_dim=3,k=10,h_dims=[64,64],actv=nn.ReLU(),sig_max=1,
                    mu_min=-3,mu_max=+3).to(device)

method = get_method(args.query_method)
if method is not None:
    id_eval = eval_ood_mdn(model,test_e_iter,'cuda')[method]
    ood_eval = eval_ood_mdn(model,test_n_iter,'cuda')[method]
    auroc, aupr = measure(id_eval,ood_eval)
    print("AUROC: [%.3f] AUPR: [%.3f]"%(auroc,aupr))