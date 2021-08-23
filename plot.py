import matplotlib.pyplot as plt
import os,json
import argparse
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--id', type=int,default=1,help='id')
parser.add_argument('--query_step', type=int,default=10,help='query step')
parser.add_argument('--exp_case', nargs='+', type=int,default=[1,2,3],help='expert case')

args = parser.parse_args()

method = ['epistemic','aleatoric','pi_entropy','random']
case_name = args.exp_case

test_l2={}
train_l2 = {}
ood = {}
ind = {}
auroc = {}
aupr = {}
for m in method:
    DIR = './res/mdn_{}/{}_{}/log.json'.format(m,args.exp_case,args.id)
    with open(DIR) as f:
        data = json.load(f)
    test_l2[m] = data['test_l2']
    train_l2[m] = data['train_l2']
    auroc[m] = np.asarray(data['auroc']).T
    aupr[m] = np.asarray(data['aupr']).T
    temp1 = data['id_eval']
    temp2 = data['ood_eval']
    key = str(args.query_step)
    ood[m] = temp2[key]
    ind[m]= temp1[key]
    case = np.asarray(data['neg_case'])
f.close()

color = [['lightcoral','oodianred','firebrick'],['palegreen','limegreen','seagreen']
        ,['skyblue','royalblue','mediumblue'],['plum','hotpink','purple']]

'''
Plot NLL, AUROC, AUPR
'''

x_range = [i for i in range(args.query_step)]
plt.figure(figsize=(8,12))
grid = plt.GridSpec(4,2)
plt.suptitle("CASE {} MDN Active Learning Result".format(case_name))
plt.subplot(grid[0,0:2])
plt.title("L2")
for i,m in enumerate(method):
    #plt.plot(test_acc[m],label=m)
    plt.plot(test_l2[m],label=m,marker='o',markersize=3,color=color[i][-1])
plt.xlabel("Query Step")
plt.ylabel("L2")
leg = plt.legend()
leg.set_title('Query Method')
plt.xticks(x_range)

for i,a in enumerate(['epis_','alea_','pi_entropy_']):
    plt.subplot(4,2,2*i+3)
    plt.title("AUROC over Query Step \nEval Method: {}".format(a))
    for j,m in enumerate(method):
        plt.plot(auroc[m][i],marker='o',markersize=3,color=color[j][2])
    plt.ylim([0.0,1])
    plt.subplot(4,2,2*i+4)
    plt.title("AUPR over Query Step\n Eval Method: {}".format(a))
    for j,m in enumerate(method):
        plt.plot(aupr[m][i],marker='o',markersize=3,color=color[j][2])
    plt.ylim([0.0,1])
plt.tight_layout()
plt.savefig("./res/mdn_{}_{}.png".format(case_name,args.id))
#plt.show()

'''
Plot Histogram by case
'''
plt.figure(figsize=(10,10))
plt.suptitle("CASE {} Histogram".format(case_name))
for i,m in enumerate(method):
    for k,j in enumerate(['epis_','alea_','pi_entropy_']):
        plt.subplot(4,3,3*i+k+1)
        plt.title("Query Method: {} \nEval Method: {}".format(m,j[:-1]))
        ood_ar = np.asarray(ood[m][j])
        case1 = np.where(case==1)[0]
        case1 = ood_ar[case1]
        case2 = np.where(case==2)[0]
        case2 = ood_ar[case2]
        case3 = np.where(case==3)[0]
        case3 = ood_ar[case3]
        id_ar = np.asarray(ind[m][j])
        # print(np.mean(id_ar),np.mean(case1),np.mean(case2),np.mean(case3))
        plt.hist(id_ar.tolist(),label='Expert', color='b',alpha=0.5)
        plt.hist(case3.tolist(),label='straight road accident',color='orangered', alpha=0.5)
        plt.hist(case1.tolist(),label='unstable',color='r', alpha=0.5)
        plt.hist(case2.tolist(),label='cross road accident',color='purple', alpha=0.5)
        if k==2:
            plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig("./res/mdn_hist_{}_{}.png".format(case_name,args.id))