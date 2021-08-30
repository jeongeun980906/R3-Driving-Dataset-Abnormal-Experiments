from tools.mixquality import  MAX_N_OBJECTS, load_expert_dataset,load_negative_dataset
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int,default=1,help='id')
parser.add_argument('--n_object', type=int,default=None,help='number of max object')
args = parser.parse_args()

if args.n_object==None:
    MAX_N_OBJECTS = 5
    dsize =35
else:
    MAX_N_OBJECTS = args.n_object
    dsize = 5 + 6*args.n_object

exp_path = "./dataset/mixquality/exp/"
neg_path = "./dataset/mixquality/neg/"
exp_case=[1,2,3]
data = torch.empty((0,dsize))
case = torch.empty((0))
mean = []
std = []
max_ = []
min_ = []

a,b,_ = load_negative_dataset(neg_path)
a = torch.cat((a,b),1)
t = torch.empty((a.size(0))).fill_(0)
temp_mean = torch.mean(a,dim=0).numpy().tolist()
temp_std = torch.std(a,dim=0).numpy().tolist()
ma = torch.max(a,dim=0)[0].numpy().tolist()
mi = torch.min(a,dim=0)[0].numpy().tolist()
mean.append(temp_mean)
std.append(temp_std)
max_.append(ma)
min_.append(mi)
case = torch.cat((case,t),dim=0)
data = torch.cat((data,a),dim=0)

for c in exp_case:
    a,b,t = load_expert_dataset(exp_path)
    a = torch.cat((a,b),1)
    temp_mean = torch.mean(a,dim=0).numpy().tolist()
    ma = torch.max(a,dim=0)[0].numpy().tolist()
    mi = torch.min(a,dim=0)[0].numpy().tolist()
    temp_std = torch.std(a,dim=0).numpy().tolist()
    mean.append(temp_mean)
    std.append(temp_std)
    max_.append(ma)
    min_.append(mi)
    case = torch.cat((case,t),dim=0)
    data = torch.cat((data,a),dim=0)

CASE_NAME = ['neg','FMTC','highway','road']
data_name = ['y','decision','deviation']
color = ['r','g','b','purple']

for n in range(MAX_N_OBJECTS):
    data_name.append('obj{}_x'.format(n+1))
    data_name.append('obj{}_y'.format(n+1))
    data_name.append('obj{}_theta'.format(n+1))
    data_name.append('obj{}_v'.format(n+1))
    data_name.append('obj{}_ax'.format(n+1))
    data_name.append('obj{}_omega'.format(n+1))
data_name.append('ax')
data_name.append('omega')

plot_size = len(data_name)//9 + 3

plt.figure(figsize=(int(plot_size*2)+12,int(plot_size*1.5)+5))
plt.suptitle("N Object: {}".format(args.n_object),fontsize=plot_size*3+5)
for j,n in enumerate(data_name):
    plt.subplot(plot_size,9,j+1)
    for i in range(4):
        x = [-0.4*(i-1.5)]
        plt.errorbar(x,mean[i][j],yerr=std[i][j],fmt='.',label=CASE_NAME[i],color=color[i])
        # plt.errorbar(x,mean[i][j],yerr=(min_[i][j],max_[i][j]),xerr=0,fmt='.',label=CASE_NAME[i])
    x = [0]
    plt.xticks(x,[n])
    plt.xlim(-1,1)
    if len(data_name)<8:
        if j==len(data_name)-1:
            plt.legend(title='Expert Case',bbox_to_anchor=(1.05, 1), loc='upper left')
    if j==4:
        plt.title("mean and variance",fontsize=plot_size*3+2)
    if j==8:
        plt.legend(title='Expert Case',bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
grid = plt.GridSpec(plot_size,11)
# '''
# tsne (S,A)
# '''
# data1 = data.numpy()
# case1 = case.numpy()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(data1)
# grid = plt.GridSpec(10,11)
# plt.subplot(grid[plot_size-6:plot_size-4,2:-2])
# plt.title("TSNE Result - (S,A)",fontsize=plot_size*3+2)
# sns.scatterplot(
#     x=tsne_results[:,0], y=tsne_results[:,1],
#     hue=case,
#     palette=sns.color_palette("hls", 4),
#     legend="full",
#     alpha=0.3
# )
# plt.tight_layout()
# '''
# TSNE - (S,S')
# '''
# data2 = data[:,:-2].numpy()
# case2 = case.numpy()
# s2 = np.zeros_like(data2)
# s2[:-1] = data2[1:].copy()
# data2 = np.concatenate((data2,s2),axis=1)
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(data2)
# plt.subplot(grid[plot_size-4:plot_size-2,2:-2])
# plt.title("TSNE Result - (S,S')",fontsize=plot_size*3+2)
# sns.scatterplot(
#     x=tsne_results[:,0], y=tsne_results[:,1],
#     hue=case,
#     palette=sns.color_palette("hls", 4),
#     legend="full",
#     alpha=0.3
# )
plt.tight_layout()
'''
TSNE - (S)
'''
data3 = data[:,:-2].numpy()
case3 = case.numpy()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data3)
plt.subplot(grid[plot_size-2:,2:-2])
plt.title("TSNE Result - (S)",fontsize=plot_size*3+2)
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=case,
    palette=sns.color_palette("hls", 4),
    legend="full",
    alpha=0.3
)
plt.savefig('data_{}.png'.format(args.id))