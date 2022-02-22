import json
path = './logger.json'

with open(path,'r') as jf:
    datas = json.load(jf)
for data in datas:
    path = data['exp_path']
    print(len(path))
    a = data['exp_unct']
    print(len(a))
    path = data['neg_path']
    print(len(path))
    a = data['neg_unct']
    print(len(a))