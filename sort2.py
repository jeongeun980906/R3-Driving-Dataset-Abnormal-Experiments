import json
from collections import OrderedDict
with open('./save.json','r') as jf:
    data = json.load(jf)
epis = data[0]
recon = data[2]
neg_path = epis['neg_path']
exp_path = epis['exp_path']
exp_epis = epis['exp_unct']
neg_epis = epis['neg_unct']
exp_recon = recon['exp_unct']
neg_recon = recon['neg_unct']
print(len(exp_recon),len(exp_epis),len(exp_path))

exp_res = {}
neg_res = {}
for p in exp_path:
    p = p.split("/")
    path = p[-4]+'_'+p[-3]
    exp_res[path] = {}
for p in neg_path:
    p = p.split("/")
    path = p[-4]+'_'+p[-3]
    neg_res[path] = {}
for p,ep,re in zip(exp_path,exp_epis,exp_recon):
    p = p.split("/")
    temp = p[-1]
    path = p[-4]+'_'+p[-3]
    data = {'epis':ep, 'recon': re}
    exp_res[path][temp] = data

for p,ep,re in zip(neg_path,neg_epis,neg_recon):
    p = p.split("/")
    temp = p[-1]
    path = p[-4]+'_'+p[-3]
    data = {'epis':ep, 'recon': re}
    neg_res[path][temp] = data

for path in exp_res:
    data = exp_res[path]
    data = sorted(data.items())
    with open('./res/files/{}.json'.format(path),'w') as jf:
        json.dump(data,jf,indent=4)

for path in neg_res:
    data = neg_res[path]
    data = sorted(data.items())
    with open('./res/files/{}.json'.format(path),'w') as jf:
        json.dump(data,jf,indent=4)
# with open('./save2.json','w') as jf:
#     json.dump(res,jf,indent=4)