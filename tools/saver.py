import json
from typing import List, Dict

class LOGGER():
    '''
    save type
    LIST [
    Dict {
        neg_path: List
        exp_path: List
        frame : Int
        method: str
        neg_unct: List
        exp_unct: List
        }
    ]
    '''
    def __init__(self,frame,method,exp_path,neg_path):
        self.frame = frame
        self.method = method
        self.path = './logger.json'
        self.neg_path = neg_path
        self.exp_path = exp_path

    def load(self):
        try:
            with open(self.path,'r') as jf:
                data = json.load(jf)
            self.data = data # List
        except:
            self.data = []

    def append(self,exp_unct,neg_unct):
        data  ={
            'frame':self.frame,
            'method': self.method,
            'exp_path':self.exp_path,
            'neg_path': self.neg_path,
            'exp_unct':exp_unct,
            'neg_unct':neg_unct
        }
        self.data.append(data)
        with open(self.path, 'w') as jf:
            json.dump(self.data,jf)