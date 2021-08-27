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
                ,'0729_neg_wooseok_50_2'
                ,'0813_exp_jeongwoo_road_1'
                ,'0813_exp_jeongwoo_road_2'
                ,'0815_exp_jeongwoo_highway_1'
                ,'0815_exp_jeongwoo_highway_2']
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
            (170,220),
            (1400,4400),
            (14000,17000),
            (1400,4400),
            (5500,8500)]

MAX_N_OBJECTS = 5
N = 96
#exp_list = [i for i in range(0,8)]
# exp_list = [2,6] #Road [0,3,4,7] #FMTC [1,5] #Highway
t_exp_list = [[0,3,4,7],[1,5,98,99],[2,6,96,97]]
neg_list = [i for i in range(8,96)]

def check_exp_case(c):
    if c in t_exp_list[0]:
        return 1
    elif c in t_exp_list[1]:
        return 2
    elif c in t_exp_list[2]:
        return 3

def load_expert_dataset(exp_path,exp_case,N_OBJECTS=None):
    """
    return
    dataset : N * STATE_DIM
    N means # of data
    STATE_DIM means dimension of state (5 + N_OBJECTS * 6)
    """
    exp_list=[]
    for i in exp_case:
        exp_list += t_exp_list[i-1]
    rt = []
    act = []
    case = []
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
            if N_OBJECTS!=None:
                data = []
                data_act = []
                n_objects = len(state['objects'])
                if n_objects==N_OBJECTS:
                    data.append(state['v'])
                    data_act.append(state['ax'])
                    data_act.append(state['omega'])
                    data.append(state['decision'])
                    data.append(state['deviation'])
                    for i in range(N_OBJECTS):
                        obj = state['objects'][i]
                        data.append(obj['x'])
                        data.append(obj['y'])
                        data.append(obj['theta'])
                        data.append(obj['v'])
                        data.append(obj['ax'])
                        data.append(obj['omega'])
                    rt.append(data)
                    act.append(data_act)
                    case.append(check_exp_case(data_index))
            else:
                data = [] 
                data_act = [] 
                data.append(state['v']) 
                data_act.append(state['ax']) 
                data_act.append(state['omega']) 
                data.append(state['decision']) 
                data.append(state['deviation']) 
                n_objects = len(state['objects']) 
                x = 0 
                y = 0 
                theta =0 
                v = 0 
                ax = 0 
                omega = 0 
                try:
                    for i in range(n_objects): 
                        obj = state['objects'][i] 
                        x += obj['x'] 
                        y += obj['y'] 
                        theta += obj['theta'] 
                        v += obj['v'] 
                        ax += obj['ax'] 
                        omega += obj['omega'] 
                    x = x/n_objects 
                    y = y/n_objects 
                    theta = theta/n_objects 
                    v = v/n_objects 
                    ax = ax/n_objects 
                    omega = omega/n_objects 
                except:
                    pass
                data.append(x) 
                data.append(y) 
                data.append(theta) 
                data.append(v) 
                data.append(ax) 
                data.append(omega) 
                # data = []
                # data_act = []
                # data.append(state['v'])
                # data_act.append(state['ax'])
                # data_act.append(state['omega'])
                # data.append(state['decision'])
                # data.append(state['deviation'])
                # n_objects = len(state['objects'])
                # for i in range(MAX_N_OBJECTS):
                #     if i < n_objects:
                #         obj = state['objects'][i]
                #         data.append(obj['x'])
                #         data.append(obj['y'])
                #         data.append(obj['theta'])
                #         data.append(obj['v'])
                #         data.append(obj['ax'])
                #         data.append(obj['omega'])
                #     else:
                #         # add dummy
                #         data.append(0)
                #         data.append(0)
                #         data.append(0)
                #         data.append(0)
                #         data.append(0)
                #         data.append(0)
                rt.append(data)
                act.append(data_act)
                case.append(check_exp_case(data_index))
    return torch.FloatTensor(rt), torch.FloatTensor(act),torch.FloatTensor(case)

def load_negative_dataset(neg_path,N_OBJECTS=None):
    """
    return
    dataset : N * STATE_DIM
    N means # of data
    STATE_DIM means dimension of state (5 + N_OBJECTS * 6)
    """
    rt = []
    act = []
    case = []
    for data_index in neg_list:
        data_name = data_name_list[data_index]
        data_path = neg_path + data_name + "/"
        state_path = data_path + "state/"
        CASE_NUM = int(data_name.split('_')[3])
        st = seq_list[data_index][0]
        en = seq_list[data_index][1]
        for seq in range(st + 1, en + 1):
            state_file = state_path + str(seq).zfill(6) + ".json"
            with open(state_file, "r") as st_json:
                state = json.load(st_json)
            if N_OBJECTS!=None:
                data = []
                data_act = []
                n_objects = len(state['objects'])
                if n_objects==N_OBJECTS:
                    data.append(state['v'])
                    data_act.append(state['ax'])
                    data_act.append(state['omega'])
                    data.append(state['decision'])
                    data.append(state['deviation'])
                    for i in range(N_OBJECTS):
                        obj = state['objects'][i]
                        data.append(obj['x'])
                        data.append(obj['y'])
                        data.append(obj['theta'])
                        data.append(obj['v'])
                        data.append(obj['ax'])
                        data.append(obj['omega'])
                    rt.append(data)
                    act.append(data_act)
                    case.append(check_neg_case(CASE_NUM))
            else:
                data = [] 
                data_act = [] 
                data.append(state['v']) 
                data_act.append(state['ax']) 
                data_act.append(state['omega']) 
                data.append(state['decision']) 
                data.append(state['deviation']) 
                n_objects = len(state['objects']) 
                x = 0 
                y = 0 
                theta =0 
                v = 0 
                ax = 0 
                omega = 0 
                try:
                    for i in range(n_objects): 
                        obj = state['objects'][i] 
                        x += obj['x'] 
                        y += obj['y'] 
                        theta += obj['theta'] 
                        v += obj['v'] 
                        ax += obj['ax'] 
                        omega += obj['omega'] 
                    x = x/n_objects 
                    y = y/n_objects 
                    theta = theta/n_objects 
                    v = v/n_objects 
                    ax = ax/n_objects 
                    omega = omega/n_objects 
                except:
                    pass
                data.append(x) 
                data.append(y) 
                data.append(theta) 
                data.append(v) 
                data.append(ax) 
                data.append(omega) 
                # data = []
                # data_act = []
                # data.append(state['v'])
                # data_act.append(state['ax'])
                # data_act.append(state['omega'])
                # data.append(state['decision'])
                # data.append(state['deviation'])
                # n_objects = len(state['objects'])
                # for i in range(MAX_N_OBJECTS):
                #     if i < n_objects:
                #         obj = state['objects'][i]
                #         data.append(obj['x'])
                #         data.append(obj['y'])
                #         data.append(obj['theta'])
                #         data.append(obj['v'])
                #         data.append(obj['ax'])
                #         data.append(obj['omega'])
                #     else:
                #         # add dummy
                #         data.append(0)
                #         data.append(0)
                #         data.append(0)
                #         data.append(0)
                #         data.append(0)
                #         data.append(0)
                rt.append(data)
                act.append(data_act)
                case.append(check_neg_case(CASE_NUM))
    
    return torch.FloatTensor(rt), torch.FloatTensor(act), torch.FloatTensor(case)

def check_neg_case(e):
    '''
    0 ~ 15: Unstable 1
    28 ~ 37: Cross Road Accident 2
    Others: Straight Road Accident 3
    '''
    if e<16:
        return 4
    elif e>27 and e<38:
        return 5
    else:
        return 6

torch.manual_seed(0)

class MixQuality():
    def __init__(self,root = "./dataset/mixquality/",exp_case=[1,2,3],train=True,neg=False,norm=True,N_OBJECTS=None):
        exp_path = root + "exp/"
        neg_path = root + "neg/"
        self.train=train
        self.neg = neg
        self.e_in, self.e_target,self.e_case = load_expert_dataset(exp_path,exp_case,N_OBJECTS)
        self.n_in, self.n_target,self.n_case = load_negative_dataset(neg_path,N_OBJECTS)
        self.e_size = self.e_in.size(0)
        self.n_size = self.n_in.size(0)
        self.norm = norm

        if self.norm:
            self.mean_in = torch.mean(torch.cat((self.e_in,self.n_in),dim=0),dim=0)
            self.std_in = torch.std(torch.cat((self.e_in,self.n_in),dim=0),dim=0)
            self.mean_t = torch.mean(torch.cat((self.e_target,self.n_target),dim=0),dim=0)
            self.std_t = torch.std(torch.cat((self.e_target,self.n_target),dim=0),dim=0)
        else:
            self.mean_in = torch.mean(torch.cat((self.e_in,self.n_in),dim=0))
            self.std_in = torch.std(torch.cat((self.e_in,self.n_in),dim=0))
            self.mean_t = torch.mean(torch.cat((self.e_target,self.n_target),dim=0))
            self.std_t = torch.std(torch.cat((self.e_target,self.n_target),dim=0))
        self.load()
        self.normaize()

    def load(self):
        rand_e_idx = torch.randperm(self.e_size)
        # rand_n_idx = torch.randperm(self.n_size)
        if self.train:
            e_idx = rand_e_idx[:int(self.e_size*0.8)] #8000
            # n_idx = rand_n_idx[:4500]
            e_in = self.e_in[e_idx]
            e_target = self.e_target[e_idx]
            # n_in = self.n_in[n_idx]
            # n_target = self.n_target[n_idx]
            # case = self.n_case[n_idx]
            self.e_label = e_idx.size(0)
            self.x = e_in
            self.y = e_target
            self.case = self.e_case[e_idx]
            # self.x = torch.cat((e_in,n_in),dim=0)
            # self.y = torch.cat((e_target,n_target),dim=0)
            # self.is_expert = torch.cat((torch.ones_like(e_idx),torch.zeros_like(n_idx)),dim=0)
            # self.case = torch.cat((torch.zeros_like(e_idx),case),dim=0)
        else:
            e_idx = rand_e_idx[int(self.e_size*0.8):]
            # n_idx = rand_n_idx[4500:]
            if not self.neg:
                self.x = self.e_in[e_idx]
                self.y = self.e_target[e_idx]
                self.case = self.e_case[e_idx]
                #self.is_expert = torch.ones_like(e_idx)
            else:
                self.x = self.n_in
                self.y = self.n_target
                self.case = self.n_case
                # self.x = self.n_in[n_idx]
                # self.y = self.n_target[n_idx]
                # self.case = self.n_case[n_idx]
                #self.is_expert = torch.zeros_like(n_idx)
            self.e_label = e_idx.size(0)

    def normaize(self):
        if self.norm:
            self.x = (self.x - self.mean_in)/(self.std_in)
            self.y = (self.y - self.mean_t)/(self.std_t)
            self.x[self.x != self.x] = 0
            self.y[self.y != self.y] = 0
        else:
            self.x = self.x.sub_(self.mean_in).div_(self.std_in)
            self.y = self.y.sub(self.mean_t).div_(self.std_t)
        #print(self.x,self.y)

if __name__ == '__main__':
    m = MixQuality(root='../dataset/mixquality/',train=False,neg=True)
    print(m.y[:100])
