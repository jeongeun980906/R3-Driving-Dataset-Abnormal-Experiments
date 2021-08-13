import json
import numpy as np
import torch

data_name_list = ['0729_exp_gunmin_FMTC'
                ,'0729_exp_gunmin_highway'
                ,'0729_exp_gunmin_road'
                ,'0729_exp_jeongwoo_FMTC'
                ,'0729_exp_sumin_FMTC'
                ,'0729_exp_sumin_highway'
                ,'0729_exp_sumin_road'
                ,'0729_exp_wooseok_FMTC'
                ,'0729_neg_gunmin_01_1'
                ,'0729_neg_gunmin_02_1'
                ,'0729_neg_gunmin_03_1'
                ,'0729_neg_gunmin_05_1'
                ,'0729_neg_gunmin_06_1'
                ,'0729_neg_gunmin_08_1'
                ,'0729_neg_gunmin_09_1'
                ,'0729_neg_gunmin_10_1'
                ,'0729_neg_gunmin_16_1'
                ,'0729_neg_gunmin_16_2'
                ,'0729_neg_gunmin_28_1'
                ,'0729_neg_gunmin_28_2'
                ,'0729_neg_gunmin_29_1'
                ,'0729_neg_gunmin_30_1'
                ,'0729_neg_gunmin_30_2'
                ,'0729_neg_gunmin_31_1'
                ,'0729_neg_gunmin_31_2'
                ,'0729_neg_gunmin_34_1'
                ,'0729_neg_gunmin_34_2'
                ,'0729_neg_gunmin_35_1'
                ,'0729_neg_gunmin_35_2'
                ,'0729_neg_gunmin_36_1'
                ,'0729_neg_gunmin_36_2'
                ,'0729_neg_gunmin_37_1'
                ,'0729_neg_gunmin_37_2'
                ,'0729_neg_gunmin_38_1'
                ,'0729_neg_gunmin_38_2'
                ,'0729_neg_gunmin_38_3'
                ,'0729_neg_gunmin_38_4'
                ,'0729_neg_gunmin_38_5'
                ,'0729_neg_gunmin_38_6'
                ,'0729_neg_gunmin_50_1'
                ,'0729_neg_gunmin_50_2'
                ,'0729_neg_jeongwoo_01_1'
                ,'0729_neg_jeongwoo_02_1'
                ,'0729_neg_jeongwoo_03_1'
                ,'0729_neg_jeongwoo_05_1'
                ,'0729_neg_jeongwoo_06_1'
                ,'0729_neg_jeongwoo_08_1'
                ,'0729_neg_jeongwoo_09_1'
                ,'0729_neg_jeongwoo_10_1'
                ,'0729_neg_jeongwoo_50_1'
                ,'0729_neg_jeongwoo_50_2'
                ,'0729_neg_sumin_01_1'
                ,'0729_neg_sumin_02_1'
                ,'0729_neg_sumin_03_1'
                ,'0729_neg_sumin_05_1'
                ,'0729_neg_sumin_06_1'
                ,'0729_neg_sumin_08_1'
                ,'0729_neg_sumin_09_1'
                ,'0729_neg_sumin_10_1'
                ,'0729_neg_sumin_38_1'
                ,'0729_neg_sumin_38_2'
                ,'0729_neg_sumin_38_3'
                ,'0729_neg_sumin_38_4'
                ,'0729_neg_sumin_42_1'
                ,'0729_neg_sumin_42_2'
                ,'0729_neg_sumin_42_3'
                ,'0729_neg_sumin_42_4'
                ,'0729_neg_sumin_50_1'
                ,'0729_neg_sumin_50_2'
                ,'0729_neg_wooseok_01_1'
                ,'0729_neg_wooseok_02_1'
                ,'0729_neg_wooseok_03_1'
                ,'0729_neg_wooseok_05_1'
                ,'0729_neg_wooseok_06_1'
                ,'0729_neg_wooseok_08_1'
                ,'0729_neg_wooseok_09_1'
                ,'0729_neg_wooseok_10_1'
                ,'0729_neg_wooseok_28'
                ,'0729_neg_wooseok_28_1'
                ,'0729_neg_wooseok_29_1'
                ,'0729_neg_wooseok_29_2'
                ,'0729_neg_wooseok_30_1'
                ,'0729_neg_wooseok_30_2'
                ,'0729_neg_wooseok_31_1'
                ,'0729_neg_wooseok_31_2'
                ,'0729_neg_wooseok_34_2'
                ,'0729_neg_wooseok_35_1'
                ,'0729_neg_wooseok_35_2'
                ,'0729_neg_wooseok_36_1'
                ,'0729_neg_wooseok_36_2'
                ,'0729_neg_wooseok_37_1'
                ,'0729_neg_wooseok_37_2'
                ,'0729_neg_wooseok_46'
                ,'0729_neg_wooseok_47'
                ,'0729_neg_wooseok_50_1'
                ,'0729_neg_wooseok_50_2']


seq_list = [(50,2450),
            (6000,9000),
            (5000,8000),
            (0,2200),
            (150,2550),
            (500,3500),
            (6000,9000),
            (50,1450),
            (40,140),
            (0,140),
            (0,160),
            (90,150),
            (90,170),
            (50,150),
            (90,190),
            (30,80),
            (80,180),
            (220,260),
            (200,220),
            (120,140),
            (120,140),
            (130,160),
            (180,200),
            (160,180),
            (145,165),
            (120,140),
            (90,110),
            (140,160),
            (180,200),
            (120,140),
            (180,200),
            (80,100),
            (120,130),
            (120,140),
            (150,170),
            (140,160),
            (150,170),
            (240,270),
            (180,200),
            (100,200),
            (80,200),
            (0,150),
            (0,150),
            (0,100),
            (80,190),
            (110,190),
            (50,150),
            (170,220),
            (70,100),
            (90,160),
            (90,290),
            (80,150),
            (70,160),
            (40,140),
            (40,80),
            (230,270),
            (50,150),
            (50,70),
            (130,150),
            (180,200),
            (120,140),
            (155,175),
            (120,140),
            (190,210),
            (140,160),
            (110,130),
            (130,150),
            (80,250),
            (80,180),
            (80,160),
            (50,150),
            (50,130),
            (120,190),
            (100,150),
            (40,110),
            (300,350),
            (360,400),
            (160,190),
            (125,155),
            (120,140),
            (100,110),
            (80,100),
            (90,110),
            (70,90),
            (110,130),
            (130,150),
            (90,110),
            (90,110),
            (230,250),
            (80,100),
            (100,130),
            (110,140),
            (170,190),
            (160,190),
            (30,130),
            (170,220)]

MAX_N_OBJECTS = 5
N = 96
exp_list = [i for i in range(0,8)]
neg_list = [i for i in range(8,96)]

def load_expert_dataset(exp_path):
    """
    return
    dataset : N * STATE_DIM
    N means # of data
    STATE_DIM means dimension of state (5 + MAX_N_OBJECTS * 6)
    """
    rt = []
    act = []
    for data_index in exp_list:
        data_name = data_name_list[data_index]
        data_path = exp_path + data_name + "/"
        state_path = data_path + "state/"

        st = seq_list[data_index][0]
        en = seq_list[data_index][1]
        for seq in range(st + 1, en + 1):
            state_file = state_path + str(seq).zfill(6) + ".json"
            with open(state_file, "r") as st_json:
                state = json.load(st_json)
            data = []
            data_act = []
            data.append(state['v'])
            data_act.append(state['ax'])
            data_act.append(state['omega'])
            data_act.append(state['decision'])
            data.append(state['deviation'])
            n_objects = len(state['objects'])
            for i in range(MAX_N_OBJECTS):
                if i < n_objects:
                    obj = state['objects'][i]
                    data.append(obj['x'])
                    data.append(obj['y'])
                    data.append(obj['theta'])
                    data.append(obj['v'])
                    data.append(obj['ax'])
                    data.append(obj['omega'])
                else:
                    # add dummy
                    data.append(0)
                    data.append(0)
                    data.append(0)
                    data.append(0)
                    data.append(0)
                    data.append(0)
            rt.append(data)
            act.append(data_act)
    
    return torch.FloatTensor(rt), torch.FloatTensor(act)

def load_negative_dataset(neg_path):
    """
    return
    dataset : N * STATE_DIM
    N means # of data
    STATE_DIM means dimension of state (5 + MAX_N_OBJECTS * 6)
    """
    rt = []
    act = []

    for data_index in neg_list:
        data_name = data_name_list[data_index]
        data_path = neg_path + data_name + "/"
        state_path = data_path + "state/"

        st = seq_list[data_index][0]
        en = seq_list[data_index][1]
        for seq in range(st + 1, en + 1):
            state_file = state_path + str(seq).zfill(6) + ".json"
            with open(state_file, "r") as st_json:
                state = json.load(st_json)
            data = []
            data_act = []
            data.append(state['v'])
            data_act.append(state['ax'])
            data_act.append(state['omega'])
            data_act.append(state['decision'])
            data.append(state['deviation'])
            n_objects = len(state['objects'])
            for i in range(MAX_N_OBJECTS):
                if i < n_objects:
                    obj = state['objects'][i]
                    data.append(obj['x'])
                    data.append(obj['y'])
                    data.append(obj['theta'])
                    data.append(obj['v'])
                    data.append(obj['ax'])
                    data.append(obj['omega'])
                else:
                    # add dummy
                    data.append(0)
                    data.append(0)
                    data.append(0)
                    data.append(0)
                    data.append(0)
                    data.append(0)
            rt.append(data)
            act.append(data_act)
    
    return torch.FloatTensor(rt), torch.FloatTensor(act)

torch.manual_seed(0)

class MixQuality():
    def __init__(self,root = "./dataset/mixquality/",train=True,neg=False):
        exp_path = root + "exp/"
        neg_path = root + "neg/"
        self.train=train
        self.neg = neg
        self.e_in, self.e_target = load_expert_dataset(exp_path)
        self.n_in, self.n_target = load_negative_dataset(neg_path)
        self.e_size = self.e_in.size(0)
        self.n_size = self.n_in.size(0)

        self.mean_in = torch.mean(torch.cat((self.e_in,self.n_in),dim=0))
        self.std_in = torch.std(torch.cat((self.e_in,self.n_in),dim=0))
        self.mean_t = torch.mean(torch.cat((self.e_target,self.n_target),dim=0))
        self.std_t = torch.std(torch.cat((self.e_target,self.n_target),dim=0))

        self.load()
        self.normaize()

    def load(self):
        rand_e_idx = torch.randperm(self.e_size)
        rand_n_idx = torch.randperm(self.n_size)
        if self.train:
            e_idx = rand_e_idx[:20000]
            n_idx = rand_n_idx[:4500]
            e_in = self.e_in[e_idx]
            e_target = self.e_target[e_idx]
            n_in = self.n_in[n_idx]
            n_target = self.n_target[n_idx]
            self.e_label = e_idx.size(0)
            self.x = torch.cat((e_in,n_in),dim=0)
            self.y = torch.cat((e_target,n_target),dim=0)
            self.is_expert = torch.cat((torch.ones_like(e_idx),torch.zeros_like(n_idx)),dim=0)
        else:
            e_idx = rand_e_idx[20000:]
            n_idx = rand_n_idx[4500:]
            if not self.neg:
                self.x = self.e_in[e_idx]
                self.y = self.e_target[e_idx]
                self.is_expert = torch.ones_like(e_idx)
            else:
                self.x = self.n_in[n_idx]
                self.y = self.n_target[n_idx]
                self.is_expert = torch.zeros_like(n_idx)
            self.e_label = e_idx.size(0)
    def normaize(self):
        self.x = self.x.sub_(self.mean_in).div_(self.std_in)
        self.y = self.y.sub_(self.mean_t).div_(self.std_t)

if __name__ == '__main__':
    m = MixQuality(root='../dataset/mixquality/',train=False,neg=True)
    print(m.y[:100])