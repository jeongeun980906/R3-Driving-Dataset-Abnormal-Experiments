from tools.mixquality import  load_expert_dataset,load_negative_dataset
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
exp_path = "./dataset/mixquality/exp/"
neg_path = "./dataset/mixquality/neg/"
exp_case=[1,2,3]
data = torch.empty((0,35))
case = torch.empty((0))
mean = []
std = []
max_ = []
min_ = []

a,b = load_negative_dataset(neg_path)
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
    a,b= load_expert_dataset(exp_path,[c])
    a = torch.cat((a,b),1)
    t = torch.empty((a.size(0))).fill_(c)
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
for n in range(5):
    data_name.append('obj{}_x'.format(n+1))
    data_name.append('obj{}_y'.format(n+1))
    data_name.append('obj{}_theta'.format(n+1))
    data_name.append('obj{}_v'.format(n+1))
    data_name.append('obj{}_ax'.format(n+1))
    data_name.append('obj{}_omega'.format(n+1))
data_name.append('ax')
data_name.append('omega')

plt.figure(figsize=(25,25))
for j,n in enumerate(data_name):
    plt.subplot(10,11,j+1)
    for i in range(4):
        x = [-0.4*(i-1.5)]
        plt.errorbar(x,mean[i][j],yerr=std[i][j],fmt='.',label=CASE_NAME[i],color=color[i])
        # plt.errorbar(x,mean[i][j],yerr=(min_[i][j],max_[i][j]),xerr=0,fmt='.',label=CASE_NAME[i])
    x = [0]
    plt.xticks(x,[n])
    plt.xlim(-1,1)
    if j==5:
        plt.title("mean and variance",fontsize=20)
    if j==10:
        plt.legend(title='Expert Case',bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    plt.tight_layout()
# plt.figure(figsize=(12,12))
# plt.suptitle("Data Distribution",fontsize=15)
# r = [[i for i in range(11)],[i for i in range(11,22)],[i for i in range(22,33)]]
# for k in range(3):
#     plt.subplot(5,1,k+1)
#     for i in range(4):
#         x = [j-0.2*(i-1.5) for j in r[k]]
#         plt.errorbar(x,mean[i][k*11:11*(k+1)],yerr=std[i][k*11:11*(k+1)],fmt='.',label=CASE_NAME[i])
#     x = r[k]
#     plt.xticks(x,data_name[k*11:11*(k+1)])
#     plt.legend(title='Expert Case',bbox_to_anchor=(1.05, 1), loc='upper left')
'''
tsne (S,A)
'''
data1 = data.numpy()
case1 = case.numpy()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data1)
grid = plt.GridSpec(10,11)
plt.subplot(grid[4:6,2:-2])
plt.title("TSNE Result - (S,A)",fontsize=20)
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=case,
    palette=sns.color_palette("hls", 4),
    legend="full",
    alpha=0.3
)
plt.tight_layout()
'''
TSNE - (S,S')
'''
data2 = data[:,:-2].numpy()
case2 = case.numpy()
s2 = np.zeros_like(data2)
s2[:-1] = data2[1:].copy()
data2 = np.concatenate((data2,s2),axis=1)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data2)
plt.subplot(grid[6:8,2:-2])
plt.title("TSNE Result - (S,S')",fontsize=20)
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=case,
    palette=sns.color_palette("hls", 4),
    legend="full",
    alpha=0.3
)
plt.tight_layout()
'''
TSNE - (S)
'''
data3 = data[:,:-2].numpy()
case3 = case.numpy()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data3)
plt.subplot(grid[8:,2:-2])
plt.title("TSNE Result - (S)",fontsize=20)
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=case,
    palette=sns.color_palette("hls", 4),
    legend="full",
    alpha=0.3
)

plt.tight_layout()
plt.savefig('data.png')