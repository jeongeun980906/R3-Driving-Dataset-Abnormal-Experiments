from solver import solver

from tools.utils import print_n_txt,Logger
from tools.measure_ood import measure
import torch
import random
import numpy as np

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,default='./dataset',help='root directory of the dataset')
parser.add_argument('--id', type=int,default=1,help='id')
parser.add_argument('--light', action='store_true',default=False,help='light verison')
parser.add_argument('--n_object', type=int,default=None,help='number of max object')
parser.add_argument('--exp_case', nargs='+', type=int,default=[1,2,3],help='expert case')
parser.add_argument('--gpu', type=int,default=0,help='gpu id')

parser.add_argument('--epoch', type=int,default=100,help='epoch')
parser.add_argument('--lr', type=float,default=1e-3,help='learning rate')
parser.add_argument('--batch_size', type=int,default=128,help='batch size')
parser.add_argument('--wd', type=float,default=1e-4,help='weight decay')
parser.add_argument('--dropout', type=float,default=0.25,help='dropout rate')
parser.add_argument('--lr_rate', type=float,default=0.9,help='learning rate schedular rate')
parser.add_argument('--lr_step', type=int,default=50,help='learning rate schedular rate')

parser.add_argument('--k', type=int,default=10,help='number of mixtures')
parser.add_argument('--norm', type=int,default=1,help='normalize dataset elementwise')
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

Solver = solver(args,device=device,SEED=SEED)
Solver.init_param()
# [init, total-init]

DIR = './res/mdn_{}/{}/'.format(args.exp_case,args.id)
DIR2 = './res/mdn_{}/{}/ckpt/'.format(args.exp_case,args.id)
try:
    os.mkdir('./res/mdn_{}'.format(args.exp_case))
except:
    pass

log = Logger(DIR+'log.json',exp_case =  Solver.test_e_dataset.case, neg_case = Solver.test_n_dataset.case)
method = ['epis_','alea_','pi_entropy_']
try:
    os.mkdir(DIR)
except:
    pass

try:
    os.mkdir(DIR2)
except:
    pass

txtName = (DIR+'log.txt')
f = open(txtName,'w') # Open txt file
print_n_txt(_f=f,_chars='Text name: '+txtName)
print_n_txt(_f=f,_chars=str(args))

train_l2, test_l2 = Solver.train_mdn(f)
log.train_res(train_l2,test_l2)

id_eval = Solver.eval_ood_mdn(Solver.test_e_iter,'cuda')
ood_eval = Solver.eval_ood_mdn(Solver.test_n_iter,'cuda')

auroc, aupr = {},{}
for m in method:
    temp1, temp2 = measure(id_eval[m],ood_eval[m])
    strTemp = ("%s AUROC: [%.3f] AUPR: [%.3f]"%(m[:-1],temp1,temp2))
    print_n_txt(_f=f,_chars= strTemp)
    auroc[m] = temp1
    aupr[m] = temp2

log.ood(id_eval,ood_eval,auroc,aupr)
torch.save(Solver.model,DIR2+'model.pth')
log.save()