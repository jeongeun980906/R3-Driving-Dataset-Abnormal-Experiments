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
    def __init__(self,path,init_indicies,init_ratio,neg_case=None):
        self.path = path
        self.train_l2 = []
        self.test_l2 = []
        self.auroc = []
        self.aupr = []
        self.idx = {}
        self.id_eval = {}
        self.ood_eval = {}
        self.idx[0]=init_indicies.numpy().tolist()
        self.neg_case = neg_case.numpy().tolist()
        self.ratio = []
        self.ratio.append(init_ratio)
        self.flag=1
    
    def append(self,train_l2,test_l2,new,ratio,id_eval,ood_eval,auroc,aupr):
        self.train_l2.append(train_l2)
        self.test_l2.append(test_l2)
        self.idx[self.flag]=new.numpy().tolist() # What is queried 
        self.ratio.append(ratio)
        self.id_eval[self.flag]=id_eval # OOD EVAL
        self.ood_eval[self.flag]=ood_eval
        self.flag+=1
        self.auroc.append(auroc)
        self.aupr.append(aupr)
    def save(self):
        try:
            os.remove(self.path)
        except:
            pass
        data = {}
        with open(self.path,'w') as json_file:
            data['train_l2']=self.train_l2
            data['test_l2']=self.test_l2
            data['query']= self.idx
            data['query_ratio'] = self.ratio
            data['neg_case'] = self.neg_case
            data['id_eval'] = self.id_eval
            data['ood_eval'] = self.ood_eval
            data['auroc'] = self.auroc
            data['aupr'] = self.aupr
            json.dump(data,json_file, indent=4)

def get_ratio(index,case):
    res = []
    total = index.size(0)
    for i in range(3):
        res.append((case[index]==i+1).sum().item()/total)
    return res