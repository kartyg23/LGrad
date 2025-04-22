import sys
import time
import os
import csv
import torch
from util import Logger
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np


# CUDA_VISIBLE_DEVICES=0 python eval_test8gan.py --dataroot  {Test-dir} --model_path {Model-Path}

vals = ['train']
multiclass = [0]

opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(opt.model_path).replace('.pth', '')

dataroot = opt.dataroot
print(f'Dataroot {opt.dataroot}')
print(f'Model_path {opt.model_path}')

# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))
model.cuda()
model.eval()

accs = [];aps = []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = False    # testing without resizing by default
    opt.no_crop = True    # testing without cropping by default
    acc, ap, r_acc, f_acc, r_precision, r_recall, r_f1, f_precision, f_recall, f_f1 = validate(model, opt)
    accs.append(acc);aps.append(ap)
    print('AP: {:2.2f}, Acc: {:2.2f},' \
    ' Acc (real): {:2.2f}, Acc (fake): {:2.2f},'.format(ap*100., acc*100., r_acc*100., f_acc*100.))
    print('Precision (real): {:2.2f}, Precision (fake): {:2.2f},'.format(r_precision*100., f_precision*100.))
    print('Recall (real): {:2.2f}, Recall (fake): {:2.2f},'.format(r_recall*100., f_recall*100.))
    print(' f1_score (real): {:2.2f}, f1_score (fake): {:2.2f}'.format(r_f1*100., f_f1*100.))
# print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

