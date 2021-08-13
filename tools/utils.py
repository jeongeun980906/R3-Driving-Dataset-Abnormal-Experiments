import os
import json
import torch

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)

class Logger():
    def __init__(self,path,init_indicies):
        self.path = path
        self.train_acc = []
        self.test_acc = []
        self.auroc = []
        self.aupr = []
        self.idx = {}
        self.idx[0]=init_indicies.numpy().tolist()
        self.flag=1
    
    def append(self,train_acc,test_acc,new,auroc,aupr):
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)
        self.idx[self.flag]=new.numpy().tolist()
        self.flag+=1
        self.auroc.append(auroc)
        self.aupr.append(aupr)
        
    def save(self):
        data = {}
        with open(self.path,'w') as json_file:
            data['train_nll']=self.train_acc
            data['test_nll']=self.test_acc
            data['query']= self.idx
            data['auroc'] = self.auroc
            data['aupr'] = self.aupr
            json.dump(data,json_file, indent=4)

def filter_expert(query,pool):
    '''
    input: index of query
    output: index of query containing only experts
    '''
    mask = pool.basedata.e_label
    a = torch.where(query<mask)[0]
    return query[a]