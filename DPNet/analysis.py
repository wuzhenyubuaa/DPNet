#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net1  import DPNet
import re

dic_224x224={}

class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot='../experiments/model76/model-32', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        
#         for name in self.net.state_dict():
            
#             print(name)
            
        for name,parameters in self.net.named_parameters():
            
#             if name[:6] =='bkbone':
#                 print(name,':',parameters.size())
                
            if re.search('fc.2', name) != None:
                
                dic_224x224[name]=parameters
                
        np.save('dic_224x224.npy', dic_224x224)
                

   

if __name__=='__main__':
    
    root = '/userhome/wuzhenyu/data/'
    
    for path in [ 'ecssd']:
        t = Test(dataset, DPNet, root + path)
#         t.save()
        # t.show()
