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
            a = sum(temp[key][1])
            qo.append(a)
            query_size = len(temp[key][1])
    query_ood[m] = 1-np.asarray(qo)/query_size
f.close()

x_range = [i for i in range(args.query_step)]
plt.figure(figsize=(8,8))
plt.suptitle("MDN Active Learning Result")
plt.subplot(2,2,1)
plt.title("NLL")
for m in method:
    #plt.plot(test_acc[m],label=m)
    plt.plot(test_nll[m],label=m,marker='o',markersize=3)
plt.xlabel("Query Step")
plt.ylabel("NLL")
plt.legend()
plt.xticks(x_range)

plt.subplot(2,2,2)
plt.title("Queried Negative Data")
for e,m in enumerate(method):
    x_range2 = [i-0.2*(e-2) for i in range(args.query_step)]
    plt.bar(x_range2,query_ood[m],label=m,width = 0.2)
plt.legend()

plt.subplot(2,2,3)
plt.title("AUROC over Query Step")
for m in method:
    plt.plot(auroc[m],label=m,marker='o',markersize=3)
plt.legend()

plt.subplot(2,2,4)
plt.title("AUPR over Query Step")
for m in method:
    plt.plot(aupr[m],label=m,marker='o',markersize=3)
plt.legend()

plt.tight_layout()
plt.savefig("./res/mdn_{}.png".format(args.id))
#plt.show()