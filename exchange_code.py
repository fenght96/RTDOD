import os
import shutil
import torch
import pdb

trm = torch.load('./res/model_0004999.pth')
base = torch.load('./output_gate_0/model_0000000.pth')
trm_ks = list(base['model'].keys())
model = {}
for k,v in trm['model'].items():
    if 'trm' in k:
        model[k.replace('_trm','.bottom_up_trm')] = v
    else:
        model[k.replace('backbone','backbone.bottom_up_rgb')] = v
print(len(model.keys()))
base['model'] = model
print(len(trm_ks))
# pdb.set_trace()
torch.save(base, './res/model_exchange.pth')
# backbone.bottom_up_trm.res3.2.conv2.norm.running_mean
# backbone.bottom_up_trm.res5.1.conv3.norm.bias
# backbone.bottom_up_rgb.stem.conv1.norm.running_var




'''
name_list = os.listdir('/home/fht/data/pretrain_rgbt/trm')
with open('/home/fht/data/pretrain_rgbt/ImageSets/Main/trainval.txt', 'w') as f:
    for name in name_list:
        name = name.split('.')[0]
        f.write(name + '\n')
        # shutil.copy('./datasets/uav/Annotations/D_0000_000000.xml', "/home/fht/data/pretrain_rgbt/Annotations/" + name + '.xml')
'''