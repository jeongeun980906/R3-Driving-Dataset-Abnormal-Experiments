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
    def __init__(self,path,exp_case=None,neg_case=None):
        self.path=path
        self.neg_case = neg_case.numpy().tolist()
        self.exp_case = exp_case.numpy().tolist()
    
    def train_res(self,train_l2,test_l2):
        self.train_l2 = train_l2
        self.test_l2 = test_l2

    def ood(self,id_eval,ood_eval,auroc,aupr):
        self.id_eval = id_eval
        self.ood_eval = ood_eval
        self.auroc = auroc
        self.aupr = aupr

    def save(self):
        try:
            os.remove(self.path)
        except:
            pass
        data = {}
        with open(self.path,'w') as json_file:
            data['train_l2']=self.train_l2
            data['test_l2']=self.test_l2
            data['neg_case'] = self.neg_case
            data['exp_case'] = self.exp_case
            data['id_eval'] = self.id_eval
            data['ood_eval'] = self.ood_eval
            data['auroc'] = self.auroc
            data['aupr'] = self.aupr
            json.dump(data,json_file, indent=4)