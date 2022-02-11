#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
# sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset1
from net1  import DPNet
import os
# from apex import amp
from torchstat import stat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='/userhome/wuzhenyu/data/dut_tr', savepath='../experiments/model76/', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=32)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8,drop_last=True)
    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
    ## network
    net    = Network(cfg)
    net.train(True)
#     stat(net, (3,288,288))
    net.cuda()
    
    
#     total = sum([param.nelement() for param in net.parameters()])
 
#     print("Number of parameter:{}".format(total))
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
#     net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    if torch.cuda.device_count() > 1:
        # torch.distributed.init_process_group(backend="nccl")
        # net = nn.parallel.DistributedDataParallel(net)  # use multiple GPU
        net = nn.DataParallel(net)

    
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()
            out1u, out2u, out2r, out3r, out4r, out5r = net(image)
            
#             loss0  = structure_loss(pred, mask)
            loss1u = structure_loss(out1u, mask)
            loss2u = structure_loss(out2u, mask)

            loss2r = structure_loss(out2r, mask)
            loss3r = structure_loss(out3r, mask)
            loss4r = structure_loss(out4r, mask)
            loss5r = structure_loss(out5r, mask)
            loss   = (loss1u+loss2u)/2+loss2r/2+loss3r/4+loss4r/8+loss5r/16

            optimizer.zero_grad()
#             with amp.scale_loss(loss, optimizer) as scale_loss:
#                 scale_loss.backward()
            loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1u':loss1u.item(), 'loss2u':loss2u.item(), 'loss2r':loss2r.item(), 'loss3r':loss3r.item(), 'loss4r':loss4r.item(), 'loss5r':loss5r.item()}, global_step=global_step)
            if step%1 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        if (epoch+1)==32:
            torch.save(net.module.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset1, DPNet)