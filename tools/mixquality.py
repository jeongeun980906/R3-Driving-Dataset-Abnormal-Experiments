import json
import torch
import numpy as np

exp_data_name_list = ['expert/scenario_%03d'%(i) for i in range(1,14)]
neg_data_name_list = ['abnormal/scenario_%03d'%(i) for i in range(1,370)]

MAX_N_OBJECTS = 5
def load_expert_dataset(path,frame):
    """
    return
    dataset : N * STATE_DIM
    N means # of data
    STATE_DIM means dimension of state (5 + N_OBJECTS * 6)
    """
    rt = []
    act = []
    case = []
    file = []
    for data_path in exp_data_name_list:
        data_info_path = path + data_path + "/summary.json"
        state_path = path + data_path + "/data/"
        with open(data_info_path, "r") as info_json:
            info = json.load(info_json)
        en = info['n_frames']
        location = info["location"]
        c = [location['highway'],location['urban'],location['FMTC']]
        for seq in range(1, en + 1 - frame):
            data = []
            data_act = []
            state_file_index = state_path + str(seq).zfill(6) + ".json"
            for it in range(frame):
                state_file = state_path + str(seq+it).zfill(6) + ".json"
                with open(state_file, "r") as st_json:
                    state = json.load(st_json)
                if it == frame-1:
                    data_act.append(state['ax'])
                    data_act.append(state['omega'])
                else:
                    data.append(state['ax'])
                    data.append(state['omega'])
                data.append(state['v'])
                data.append(state['decision'])
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
            case.append(c)
            file.append(state_file_index)
    return torch.FloatTensor(rt), torch.FloatTensor(act),torch.FloatTensor(case), np.asarray(file)

def load_negative_dataset(path,frame):
    """
    return
    dataset : N * STATE_DIM
    N means # of data
    STATE_DIM means dimension of state (5 + N_OBJECTS * 6)
    """
    rt = []
    act = []
    case = []
    file = []
    for data_path in neg_data_name_list:
        data_info_path = path + data_path + "/summary.json"
        state_path = path + data_path + "/data/"
        with open(data_info_path, "r") as info_json:
            info = json.load(info_json)
        en = info['n_frames']
        road = info['road']
        hazard = info['hazard']
        c = [road["straight"],road['cross'],hazard['unstable_driving'],
                hazard['failing_lane_keeping'],hazard['dangerous_lane_changing'],
                hazard['dangerous_overtaking'],hazard['near_collision']]
        for seq in range(1, en + 1-frame):
            data = []
            data_act = []
            state_file_index = state_path + str(seq).zfill(6) + ".json"
            for it in range(frame):
                state_file = state_path + str(seq+it).zfill(6) + ".json"
                with open(state_file, "r") as st_json:
                    state = json.load(st_json)
                if it == frame-1:
                    data_act.append(state['ax'])
                    data_act.append(state['omega'])
                else:
                    data.append(state['ax'])
                    data.append(state['omega'])
                data.append(state['v'])
                data.append(state['decision'])
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
            case.append(c)
            file.append(state_file_index)
    return torch.FloatTensor(rt), torch.FloatTensor(act), torch.FloatTensor(case),np.asarray(file)

torch.manual_seed(0)

class MixQuality():
    def __init__(self,root = "./R3-Driving-Dataset",train=True,neg=False,norm=True,exp_case=[1,2,3],frame=1):
        self.train=train
        self.neg = neg
        self.e_in, self.e_target,self.e_case, self.file_expert = load_expert_dataset(root,frame)
        self.n_in, self.n_target,self.n_case, self.file_negative = load_negative_dataset(root,frame)
        # print(self.e_in.size(),self.n_in.size(),self.e_target.size(),self.n_target.size(),frame)
        self.e_size = self.e_in.size(0)
        self.n_size = self.n_in.size(0)
        self.norm = norm
        self.mean_in = torch.mean(torch.cat((self.e_in,self.n_in),dim=0),dim=0)
        self.std_in = torch.std(torch.cat((self.e_in,self.n_in),dim=0),dim=0)
        self.mean_t = torch.mean(torch.cat((self.e_target,self.n_target),dim=0),dim=0)
        self.std_t = torch.std(torch.cat((self.e_target,self.n_target),dim=0),dim=0)

        self.load()
        self.normaize()

    def load(self):
        rand_e_idx = torch.randperm(self.e_size)
        # rand_n_idx = torch.randperm(self.n_size)
        if self.train:
            e_idx = rand_e_idx[:int(self.e_size*0.8)] # 8000
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
            self.path = self.file_expert[e_idx.numpy()]
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
                self.path = self.file_expert[e_idx.numpy()]
                #self.is_expert = torch.ones_like(e_idx)
            else:
                self.x = self.n_in
                self.y = self.n_target
                self.case = self.n_case
                self.path = self.file_negative
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
            self.max_in = torch.max(torch.cat((self.e_in,self.n_in),dim=0),dim=0)[0]
            self.min_in = torch.min(torch.cat((self.e_in,self.n_in),dim=0),dim=0)[0]
            self.max_t = torch.max(torch.cat((self.e_target,self.n_target),dim=0),dim=0)[0]
            self.min_t = torch.min(torch.cat((self.e_target,self.n_target),dim=0),dim=0)[0]
            self.x = (self.x - self.min_in)/(self.max_in-self.min_in)
            self.y = (self.y - self.min_t)/(self.max_t-self.min_t)

if __name__ == '__main__':
    m = MixQuality(root='../dataset/mixquality/',train=False,neg=True)
    print(m.y[:100])
