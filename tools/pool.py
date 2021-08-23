from tools.dataloader import total_dataset,subset_dataset,quey_dataset
import torch
import copy
import random

class AL_pool():
    def __init__(self,root='./dataset',num_init=100,exp_case=[1,2,3],norm=True):
        self.basedata=total_dataset(root=root,norm=norm,exp_case=exp_case)
        self.batch_size=128
        self.expert_size = self.basedata.e_label
        self.total_size = self.basedata.__len__()
        self.idx = torch.tensor(random.sample(range(self.expert_size), num_init))
        
    def subset_dataset(self,indices):
        '''
        Input: Top N index of acqusition function
        Output: Filtered Dataset
        '''
        self.idx = torch.cat((self.idx,indices),0)
        filter = self.basedata.e_label
        a = torch.where(self.idx<filter)[0]
        input_index = self.idx[a] # Filter the Negative Data
        x = copy.deepcopy(self.basedata.x[input_index])
        y = copy.deepcopy(self.basedata.y[input_index])
        total = torch.range(0,self.total_size-1,dtype=torch.int64)
        mask = torch.ones_like(total, dtype=torch.bool)
        mask[self.idx] = False
        self.unlabled_idx = total[mask]
        labeled_subset = subset_dataset(x,y)
        train_loader = torch.utils.data.DataLoader(labeled_subset, batch_size=self.batch_size, 
                        shuffle=False)
        infer_loader = self.get_unlabled_pool()
        return train_loader,infer_loader

    def get_unlabled_pool(self):
        print(self.unlabled_idx.size())
        x = copy.deepcopy(self.basedata.x[self.unlabled_idx])
        query_pool = quey_dataset(x)
        loader  = torch.utils.data.DataLoader(query_pool, batch_size=self.batch_size, 
                        shuffle=False)
        return loader

if __name__ == '__main__':
    p = AL_pool()
    _,_ = p.subset_dataset(torch.zeros(size=(0,1),dtype=torch.int64).squeeze(1))
    print(p.unlabled_idx[:4])
    _,_ = p.subset_dataset(p.unlabled_idx[:4])
    print(p.idx,p.unlabled_idx)