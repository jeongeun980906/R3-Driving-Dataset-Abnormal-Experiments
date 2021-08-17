import matplotlib.pyplot as plt
import os,json
import argparse
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--id', type=int,default=1,help='id')
parser.add_argument('--query_step', type=int,default=10,help='query step')

args = parser.parse_args()

method = ['epistemic','aleatoric','pi_entropy','random']

test_nll={}
train_nll = {}
query_ood = {}
auroc = {}
aupr = {}
for m in method:
    DIR = './res/mdn_{}/{}/log.json'.format(m,args.id)
    with open(DIR) as f:
        data = json.load(f)
    test_nll[m] = data['test_nll']
    train_nll[m] = data['train_nll']
    auroc[m] = data['auroc']
    aupr[m] = data['aupr']
    temp = data['query']
    qo = []
    for key in temp:
        a = int(key)
        if a>0:
            a = temp[key][1].count(1)
            b = temp[key][1].count(2)
            c = temp[key][1].count(3)
            qo.append([a,b,c])
            query_size = len(temp[key][1])
            #print(query_size)
    query_ood[m] = np.asarray(qo).T/query_size
f.close()

color = [['lightcoral','indianred','firebrick'],['palegreen','limegreen','seagreen']
        ,['skyblue','cornflowerblue','mediumblue'],['plum','hotpink','purple']]

x_range = [i for i in range(args.query_step)]
plt.figure(figsize=(8,8))
plt.suptitle("MDN Active Learning Result")
plt.subplot(2,2,1)
plt.title("NLL")
for i,m in enumerate(method):
    #plt.plot(test_acc[m],label=m)
    plt.plot(test_nll[m],label=m,marker='o',markersize=3,color=color[i][-1])
plt.xlabel("Query Step")
plt.ylabel("NLL")
plt.legend()
plt.xticks(x_range)

plt.subplot(2,2,2)
plt.title("Queried Negative Data")

for e,m in enumerate(method):
    offset=0
    x_range2 = [i-0.2*(e-2) for i in range(args.query_step)]
    for i in range(3):
        plt.bar(x_range2,query_ood[m][i],bottom=offset,width = 0.2,color=color[e][i])
        offset += query_ood[m][i]
        
plt.subplot(2,2,3)
plt.title("AUROC over Query Step")
for i,m in enumerate(method):
    plt.plot(auroc[m],marker='o',markersize=3,color=color[i][2])

plt.subplot(2,2,4)
plt.title("AUPR over Query Step")
for i,m in enumerate(method):
    plt.plot(aupr[m],marker='o',markersize=3,color=color[i][2])

plt.tight_layout()
plt.savefig("./res/mdn_{}.png".format(args.id))
#plt.show()