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
    def __init__(self,path,init_indicies,neg_case=None):
        self.path = path
        self.train_acc = []
        self.test_acc = []
        self.auroc = []
        self.aupr = []
        self.idx = {}
        self.id_eval = {}
        self.ood_eval = {}
        self.idx[0]=init_indicies.numpy().tolist()
        self.neg_case = neg_case.numpy().tolist()
        self.flag=1
    
    def append(self,train_acc,test_acc,new,id_eval,ood_eval,auroc,aupr):
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)
        self.idx[self.flag]=new.numpy().tolist() # What is queried 
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
            data['train_nll']=self.train_acc
            data['test_nll']=self.test_acc
            data['query']= self.idx
            data['neg_case'] = self.neg_case
            data['id_eval'] = self.id_eval
            data['ood_eval'] = self.ood_eval
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