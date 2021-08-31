from tools.mixquality import MixQuality
from torchvision import transforms
import torch.utils.data as data
import torch

class mixquality_dataset(data.Dataset):
    def __init__(self,root='./dataset/mixquality/',train=True,neg=False,norm=True,exp_case=[1,2,3],frame=1):
        mix = MixQuality(root=root,train=train,neg=neg,norm=norm,exp_case=exp_case,frame=frame)
        self.x = mix.x
        self.y = mix.y
        self.e_label = mix.e_label
        self.case = mix.case

    def __getitem__(self, index):
        data, target = self.x[index], self.y[index]
        return data,target
    
    def __len__(self):
        return len(self.x)

class mixquality_dataset_mha(data.Dataset):
    def __init__(self,root='./dataset/mixquality/',train=True,neg=False,norm=True,exp_case=[1,2,3],frame=1):
        mix = MixQuality_MHA(root=root,train=train,neg=neg,norm=norm,exp_case=exp_case,frame=frame)
        self.x = mix.x
        self.y = mix.y
        self.e_label = mix.e_label
        self.case = mix.case

    def __getitem__(self, index):
        data, target = self.x[index], self.y[index]
        return data,target
    
    def __len__(self):
        return len(self.x)