import matplotlib.pyplot as plt
import os,json
import argparse
import numpy as np
from tools.measure_ood import measure
import matplotlib.ticker as mticker

parser = argparse.ArgumentParser()

parser.add_argument('--id', type=int,default=1,help='id')

args = parser.parse_args()

DIR = './res/mdn/{}/log.json'.format(args.id)
with open(DIR) as f:
    data = json.load(f)
test_l2 = data['test_l2']
train_l2 = data['train_l2']
auroc = data['auroc']
aupr = data['aupr']
ind = data['id_eval']
ood = data['ood_eval']

neg_case = np.asarray(data['neg_case'])
exp_case = np.asarray(data['exp_case'])
f.close()

'''
Plot NLL, AUROC, AUPR
'''
plt.figure(figsize=(20,20))
grid = plt.GridSpec(5,6)
#plt.suptitle("MDN Learning Result")
plt.subplot(grid[0,:5])
plt.title("L2")
plt.plot(train_l2,'--k',label='train')
plt.plot(test_l2,label='test',color='b')
plt.xlabel("Epoch")
plt.ylabel("L2")
leg = plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
'''
Plot Histogram by case
Plot ROC curve in one plot
'''
situation = [[0,2],[0,3],[0,6],[1,2],[1,3],[0,4],[0,5],[1,6]]
situation_name = ['straighht_road','cross_road','unstable','lane_keeping',
                    'lane_changing','overtaking','collision']
for k,j in enumerate(['epis_','alea_','pi_entropy_']):
    ood_ar = np.asarray(ood[j])
    case4 = np.where(neg_case[:,0]==1)[0] # straight_road_index
    case4 = ood_ar[case4]
    case5 = np.where(neg_case[:,1]==1)[0] # cross_road_index
    case5 = ood_ar[case5]
    ind_ar = np.asarray(ind[j])

    case1 = np.where(exp_case[:,0]==1)[0] 
    case1 = ind_ar[case1]
    case2 = np.where(exp_case[:,1]==1)[0] 
    case2 = ind_ar[case2]
    case3 = np.where(exp_case[:,2]==1)[0] 
    case3 = ind_ar[case3]

    max_1 = np.max(ood_ar)
    min_1 = np.min(ood_ar)
    max_2 = np.max(ind_ar)
    min_2 = np.min(ind_ar)
    min_ = min(min_1,min_2)
    max_ = max(max_1,max_2)
    
    plt.subplot(grid[k+1,0])
    plt.title("\n \n %s \n"%('Expert: FMTC'))
    plt.hist(case1.tolist(),color='limegreen', alpha=0.5,orientation="horizontal")
    plt.ylim((min_,max_))
    plt.tight_layout()

    plt.subplot(grid[k+1,1])
    plt.title("\n \n %s \n"%('Expert: Urban'))
    plt.hist(case2.tolist(),color='royalblue', alpha=0.5,orientation="horizontal")
    plt.ylim((min_,max_))
    plt.tight_layout()

    plt.subplot(grid[k+1,2])
    plt.title("Eval Method: %s \nAUROC:[%.3f] AUPR: [%.3f] \n %s \n"%(j[:-1],auroc[j],aupr[j],'Expert: HigWay'))
    plt.hist(case3.tolist(),color='lightseagreen', alpha=0.5,orientation="horizontal")
    plt.ylim((min_,max_))
    plt.tight_layout()

    plt.subplot(grid[k+1,3])
    plt.title("\n \n %s \n"%('Negative: Straight Road'))
    plt.hist(case4.tolist(),color='r', alpha=0.5,orientation="horizontal")
    plt.ylim((min_,max_))
    plt.tight_layout()

    plt.subplot(grid[k+1,4])
    plt.title("\n \n %s \n"%('Negative: Cross Road'))
    plt.hist(case5.tolist(),color='orange', alpha=0.5,orientation="horizontal")
    plt.ylim((min_,max_))
    plt.tight_layout()
    
    plt.subplot(grid[4,2*k:2*k+2])
    plt.title("ROC Curve")
    for i,j in situation:
        ood_temp1 = np.where(neg_case[:,i]==1)[0]
        ood_temp2 = np.where(neg_case[:,j]==1)[0]
        ood_temp = np.intersect1d(ood_temp1,ood_temp2)
        ood_temp = ood_ar[ood_temp].tolist()
        id_temp = ind_ar.tolist()
        # print(len(ood_temp),len(id_temp))
        auroc_temp, aupr_temp, fpr,tpr = measure(id_temp,ood_temp,True)
        plt.plot(fpr,tpr,label='%s,%s\nAUROC:%.3f AUPR %.3f'%(situation_name[i],situation_name[j],auroc_temp,aupr_temp))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
plt.savefig("./res/mdn_{}.png".format(args.id))