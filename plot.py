import matplotlib.pyplot as plt
import os,json
import argparse
import numpy as np
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
plt.figure(figsize=(15,8))
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
    #for i in range(7):


plt.tight_layout()
plt.savefig("./res/mdn_{}.png".format(args.id))