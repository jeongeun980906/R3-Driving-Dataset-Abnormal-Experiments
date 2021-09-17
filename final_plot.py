import matplotlib.pyplot as plt
import os,json
import argparse
import numpy as np
from numpy.core.fromnumeric import size
from tools.measure_ood import measure

MODE = ['mdn','vae']
ID = [1,1]
method = [['epis_','alea_','pi_entropy_'],['recon_']]
auroc,aupr,ind,ood={},{},{},{}
for i,(m,idx) in enumerate(zip(MODE,ID)):
    print(m,i,idx)
    DIR = './res/{}/{}/log.json'.format(m,idx)
    with open(DIR) as f:
        data = json.load(f)
    auroc_t = data['auroc']
    aupr_t = data['aupr']
    ind_t = data['id_eval']
    ood_t = data['ood_eval']
    for m in method[i]:
        auroc[m] = auroc_t[m]
        aupr[m] = aupr_t[m]
        ind[m] = ind_t[m]
        ood[m] = ood_t[m]
    neg_case = np.asarray(data['neg_case'])
    exp_case = np.asarray(data['exp_case'])

situation = [[0,2],[0,3],[0,4],[0,5],[0,6],[1,2],[1,3],[1,6]]
situation_name = ['straight_road','crossroad','unstable','lane_keeping',
                    'lane_changing',' overtaking ','near_collision']
linestyles = ['-','-','-', '-.','-.', '--','--','--']
fig = plt.figure(figsize=(25,10)) # 15 15
#plt.suptitle("\n",fontsize=25)
#fig.text(0.25, 0.97, "frame1", fontsize=28)
fig_index=0
for a,k in enumerate(method):
    for m in k:
        ood_ar = np.asarray(ood[m])
        ind_ar = np.asarray(ind[m])
        plt.subplot(2,4,fig_index+1)
        # if m=='pi_entropy_':
        #     fig_index+=4
        # else:
        #     fig_index+=1
        fig_index+=1
        plt.title("frame1: %s\n AUROC: %.3f AUPR: %.3f"%(m[:-1],auroc[m],aupr[m]),fontsize=17)
        A=0
        for i,j in situation:
            ood_temp1 = np.where(neg_case[:,i]==1)[0]
            ood_temp2 = np.where(neg_case[:,j]==1)[0]
            ood_temp = np.intersect1d(ood_temp1,ood_temp2)
            ood_temp = ood_ar[ood_temp].tolist()
            id_temp = ind_ar.tolist()
            # print(len(ood_temp),len(id_temp))
            auroc_temp, aupr_temp, fpr,tpr = measure(id_temp,ood_temp,True)
            # plt.plot(fpr,tpr,label='%s %s\nAUROC:%.3f \nAUPR %.3f'%(situation_name[i],situation_name[j],auroc_temp,aupr_temp))
            plt.plot(fpr,tpr,label='%s %s'%(situation_name[i],situation_name[j]),linestyle=linestyles[A])
            A+=1
        #legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2,fontsize='xx-large')
        # text = legend.get_texts()
        # for i in text:
        #     i.set_fontsize(15)
        if m == 'recon_':
            plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='xx-large')
        plt.tight_layout()

ID = [2,2]
method = [['epis_','alea_','pi_entropy_'],['recon_']]
auroc,aupr,ind,ood={},{},{},{}
for i,(m,idx) in enumerate(zip(MODE,ID)):
    print(m,i,idx)
    DIR = './res/{}/{}/log.json'.format(m,idx)
    with open(DIR) as f:
        data = json.load(f)
    auroc_t = data['auroc']
    aupr_t = data['aupr']
    ind_t = data['id_eval']
    ood_t = data['ood_eval']
    for m in method[i]:
        auroc[m] = auroc_t[m]
        aupr[m] = aupr_t[m]
        ind[m] = ind_t[m]
        ood[m] = ood_t[m]
    neg_case = np.asarray(data['neg_case'])
    exp_case = np.asarray(data['exp_case'])

#fig.text(0.75, 0.97, "frame5", fontsize=28)
fig_index= 5#4
for a,k in enumerate(method):
    for m in k:
        ood_ar = np.asarray(ood[m])
        ind_ar = np.asarray(ind[m])
        plt.subplot(2,4,fig_index)
        # if m=='pi_entropy_':
        #     fig_index+=2
        # else:
        #     fig_index+=1
        fig_index+=1
        plt.title("frame5: %s\n AUROC: %.3f AUPR: %.3f"%(m[:-1],auroc[m],aupr[m]),fontsize=17)
        A=0
        for i,j in situation:
            ood_temp1 = np.where(neg_case[:,i]==1)[0]
            ood_temp2 = np.where(neg_case[:,j]==1)[0]
            ood_temp = np.intersect1d(ood_temp1,ood_temp2)
            ood_temp = ood_ar[ood_temp].tolist()
            id_temp = ind_ar.tolist()
            # print(len(ood_temp),len(id_temp))
            auroc_temp, aupr_temp, fpr,tpr = measure(id_temp,ood_temp,True)
            # plt.plot(fpr,tpr,label='%s %s\nAUROC:%.3f \nAUPR %.3f'%(situation_name[i],situation_name[j],auroc_temp,aupr_temp))
            plt.plot(fpr,tpr,label='%s %s'%(situation_name[i],situation_name[j]),linestyle=linestyles[A])
            A+=1
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2,fontsize='xx-large')
        # if m == 'recon_':
        #     plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='x-large')
        #else:
        plt.tight_layout()

plt.savefig("res.png")