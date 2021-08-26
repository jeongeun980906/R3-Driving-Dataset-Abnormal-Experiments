from tools.mixquality import MixQuality
from torchvision import transforms
import torch.utils.data as data
import torch

class total_dataset(data.Dataset):
    def __init__(self,root='./dataset',train=True,neg=False,norm=True,exp_case=[1,2,3],N_OBJECTS=None):
        mix = MixQuality(root=root+'/mixquality/',train=train,neg=neg,norm=norm,exp_case=exp_case,N_OBJECTS=N_OBJECTS)
        self.x = mix.x
        self.y = mix.y
        self.e_label = mix.e_label
        self.case = mix.case

    def __getitem__(self, index):
        '''
        only used for inference
        '''
        img, target = self.x[index], self.y[index]
        return img,target
    
    def __len__(self):
        return len(self.x)

class subset_dataset(data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        '''
        for training labled pool
        '''
        img, target = self.x[index], self.y[index]
        return img,target
        
    def __len__(self):
        return len(self.x)

class quey_dataset(data.Dataset):
    def __init__(self,x):
        self.x = x
    def __getitem__(self, index):
        '''
        only used for inference
        '''
        img = self.x[index]
        return img
        
    def __len__(self):
        return self.x.size(0)