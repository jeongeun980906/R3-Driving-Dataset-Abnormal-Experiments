from torch.functional import tensordot
from solver import solver

from tools.measure_ood import measure
import torch
import random
import numpy as np

import argparse
import os
from tools.saver import LOGGER

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,default='.',help='root directory of the dataset')
parser.add_argument('--id', type=int,default=1,help='id')
parser.add_argument('--mode', type=str,default='mdn',help='mdn vae')
parser.add_argument('--gpu', type=int,default=0,help='gpu id')
parser.add_argument('--frame', type=int,default=1,help='frame')
parser.add_argument('--exp_case', type=int, nargs='+',default=[1,2,3],help='expert case')

parser.add_argument('--epoch', type=int,default=100,help='epoch')
parser.add_argument('--lr', type=float,default=1e-3,help='learning rate')
parser.add_argument('--batch_size', type=int,default=128,help='batch size')
parser.add_argument('--wd', type=float,default=1e-4,help='weight decay')
parser.add_argument('--dropout', type=float,default=0.25,help='dropout rate')
parser.add_argument('--lr_rate', type=float,default=0.9,help='learning rate schedular rate')
parser.add_argument('--lr_step', type=int,default=50,help='learning rate schedular rate')

# Parser for MDN
parser.add_argument('--k', type=int,default=10,help='number of mixtures')
parser.add_argument('--norm', type=int,default=1,help='normalize dataset elementwise')
parser.add_argument('--sig_max', type=float,default=1,help='sig max')

# Parser for VAE
parser.add_argument('--h_dim',  type=int, nargs='+',default=[20],help='h dim for vae')
parser.add_argument('--z_dim', type=int,default=10,help='z dim for vae')

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
# [init, total-init]

if args.mode == 'mdn':
    method = 'epis_'
    
elif args.mode == 'vae':
    method = 'recon_'
else:
    raise NotImplementedError
OUT = './res/{}_{}_path.txt'.format(args.mode,args.frame)
DIR2 = './res/{}/{}/ckpt/'.format(args.mode,args.id)

saver = LOGGER(args.frame,method,
        Solver.test_e_dataset.path.tolist(),
        Solver.test_n_dataset.path.tolist())

state_dict = torch.load(DIR2+'model.pt')
Solver.model.load_state_dict(state_dict)

id_eval = Solver.eval_func(Solver.test_e_iter,'cuda')
ood_eval = Solver.eval_func(Solver.test_n_iter,'cuda')

saver.load()
saver.append(id_eval[method],ood_eval[method])
# unct_id_eval = torch.tensor(id_eval[method])
# unct_ood_eval = torch.tensor(ood_eval[method])

# id_top_k = torch.topk(unct_id_eval,k=5,largest=False).indices
# ood_top_k = torch.topk(unct_ood_eval,k=5,largest=True).indices

# id_path = Solver.test_e_dataset.path[id_top_k.numpy()]
# ood_path = Solver.test_n_dataset.path[ood_top_k.numpy()]
# print('unct',id_path,ood_path)

# data = 'Frame: {} \nExpert traj \n'.format(args.frame) 
# for ip in id_path:
#     data = data + ip + '\n'

# data += '\n Negative traj \n'
# for op in ood_path:
#     data = data + op + '\n'

# with open(OUT,'w') as f:
#     f.write(data)