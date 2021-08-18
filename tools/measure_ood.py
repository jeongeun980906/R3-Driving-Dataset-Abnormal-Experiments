from sklearn.metrics import average_precision_score,roc_auc_score
import torch
import matplotlib.pyplot as plt

def measure(id_eval,ood_eval,plot=False):
    size_id = len(id_eval)
    size_ood = len(ood_eval)
    GT=[0 for _ in range(size_id)] + [1 for _ in range(size_ood)]
    EVAL=id_eval+ood_eval
    auroc=roc_auc_score(GT,EVAL)
    aupr = average_precision_score(GT,EVAL)
    if plot:
        plt.figure()
        plt.hist(id_eval,label='id')
        plt.hist(ood_eval,label='ood')
        plt.savefig('temp.png')
    return auroc,aupr