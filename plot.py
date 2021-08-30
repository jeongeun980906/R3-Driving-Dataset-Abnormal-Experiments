import matplotlib.pyplot as plt
import os,json
import argparse
import numpy as np
from tools.measure_ood import measure
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
plt.figure(figsize=(15,15))
grid = plt.GridSpec(3,3)
plt.suptitle("MDN Learning Result")
plt.subplot(grid[0,0:3])
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
    plt.subplot(3,3,k+4)
    plt.title("Eval Method: %s \nAUROC:[%.3f] AUPR: [%.3f]"%(j[:-1],auroc[j],aupr[j]))
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

    print(np.mean(case1),np.mean(case2),np.mean(case3))
    print(np.mean(case4),np.mean(case5))

    plt.hist(case1.tolist(),label='Expert: FMTC',color='limegreen', alpha=0.5)
    plt.hist(case2.tolist(),label='Expert: Highway',color='royalblue', alpha=0.5)
    plt.hist(case3.tolist(),label='Expert: Road',color='lightseagreen', alpha=0.5)

    plt.hist(case4.tolist(),label='Negative: straight road accident',color='r', alpha=0.3)
    plt.hist(case5.tolist(),label='Negative: cross road accident',color='orange', alpha=0.3)
    if k==2:
        plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.subplot(3,3,k+7)
    plt.title("ROC Curve")
    for i,j in situation:
        ood_temp1 = np.where(neg_case[:,i]==1)[0]
        ood_temp2 = np.where(neg_case[:,j]==1)[0]
        ood_temp = np.intersect1d(ood_temp1,ood_temp2)
        ood_temp = ood_ar[ood_temp].tolist()
        id_temp = ind_ar.tolist()
        print(len(ood_temp))
        auroc_temp, aupr_temp, fpr,tpr = measure(id_temp,ood_temp,True)
        plt.plot(fpr,tpr,label='%s,%s\nAUROC:%.3f AUPR %.3f'%(situation_name[i],situation_name[j],auroc_temp,aupr_temp))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
plt.savefig("./res/mdn_{}.png".format(args.id))