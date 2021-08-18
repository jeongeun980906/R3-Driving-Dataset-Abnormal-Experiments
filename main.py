from tools.pool import AL_pool
from solver import solver
from tools.dataloader import total_dataset
from tools.utils import print_n_txt,Logger,filter_expert
from tools.measure_ood import measure
import torch
import random
import numpy as np

import argparse
import os
from MDN.eval import eval_ood_mdn

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,default='./dataset',help='root directory of the dataset')
parser.add_argument('--id', type=int,default=1,help='id')
parser.add_argument('--exp_case', nargs='+', type=int,default=[1,2,3],help='expert case')
parser.add_argument('--gpu', type=int,default=0,help='gpu id')

parser.add_argument('--query_step', type=int,default=10,help='query step')
parser.add_argument('--query_size', type=int,default=100,help='query size')
parser.add_argument('--init_pool', type=int,default=200,help='number of initial data')
parser.add_argument('--query_method', type=str,default='epistemic',help='query method')
parser.add_argument('--epoch', type=int,default=100,help='epoch')
parser.add_argument('--init_weight', type=bool,default=True,help='init weight on every query step')

parser.add_argument('--lr', type=float,default=1e-3,help='learning rate')
parser.add_argument('--batch_size', type=int,default=128,help='batch size')
parser.add_argument('--wd', type=float,default=1e-4,help='weight decay')
parser.add_argument('--dropout', type=float,default=0.25,help='dropout rate')
parser.add_argument('--lr_rate', type=float,default=0.5,help='learning rate schedular rate')
parser.add_argument('--lr_step', type=int,default=50,help='learning rate schedular rate')

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

p = AL_pool(root=args.root,num_init=args.init_pool,exp_case=args.exp_case)
torch.manual_seed(SEED)
test_e_dataset = total_dataset(root = args.root, train=False,neg=False,exp_case=args.exp_case)
test_e_iter = torch.utils.data.DataLoader(test_e_dataset, batch_size=args.batch_size, 
                        shuffle=False)
torch.manual_seed(SEED)
test_n_dataset = total_dataset(root = args.root, train=False,neg=True,exp_case=args.exp_case)
test_n_iter = torch.utils.data.DataLoader(test_n_dataset, batch_size=args.batch_size, 
                        shuffle=False)

AL_solver = solver(args,device=device)
AL_solver.init_param()

label_iter,unlabel_iter = p.subset_dataset(torch.zeros(size=(0,1),dtype=torch.int64).squeeze(1))
# [init, total-init]

DIR = './res/mdn_{}/{}/'.format(args.query_method,args.id)
DIR2 = './res/mdn_{}/{}/ckpt/'.format(args.query_method,args.id)
try:
    os.mkdir('./res/mdn_{}'.format(args.query_method))
except:
    pass

log = Logger(DIR+'log.json',p.idx,neg_case = test_n_dataset.case)
method = ['epis_','alea_','pi_entropy_']

try:
    os.mkdir(DIR)
except:
    pass

try:
    os.mkdir(DIR2)
except:
    pass

for i in range(args.query_step):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    txtName = (DIR+'{}_log.txt'.format(i))
    f = open(txtName,'w') # Open txt file
    print_n_txt(_f=f,_chars='Text name: '+txtName)
    print_n_txt(_f=f,_chars=str(args))
    unl_size = len(p.unlabled_idx)
    final_train_acc, final_test_acc = AL_solver.train_mdn(label_iter,test_e_iter,test_n_iter,f)
    id = AL_solver.query_data(unlabel_iter,unl_size)
    new = p.unlabled_idx[id]
    # filter_new = filter_expert(new,p)
    # temp =torch.where(p.basedata.case[new]>0)[0].size(0)
    # save_query = torch.cat((new.unsqueeze(0),p.basedata.case[new].unsqueeze(0)),dim=0)
    label_iter,unlabel_iter = p.subset_dataset(new)
    id_eval = eval_ood_mdn(AL_solver.model,test_e_iter,'cuda')
    ood_eval = eval_ood_mdn(AL_solver.model,test_n_iter,'cuda')
    print(len(ood_eval['alea_']))
    auroc, aupr = [],[]
    for m in method:
        temp1, temp2 = measure(id_eval[m],ood_eval[m])
        strTemp = ("%s AUROC: [%.3f] AUPR: [%.3f]"%(m[:-1],temp1,temp2))
        print_n_txt(_f=f,_chars= strTemp)
        auroc.append(temp1)
        aupr.append(temp2)
    strTemp = ("Labled size : %d Unlabled size: %d")%(p.idx.size(0),p.unlabled_idx.size(0))
    print_n_txt(_f=f,_chars= strTemp)
    log.append(final_train_acc,final_test_acc,new,id_eval,ood_eval,auroc,aupr)
    torch.save(AL_solver.model,DIR2+'{}.pth'.format(i))
log.save()