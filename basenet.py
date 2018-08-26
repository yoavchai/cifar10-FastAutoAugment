#!/usr/bin/env python

"""
    basenet.py
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from helpers import to_numpy
from lr import LRSchedule

TORCH_VERSION_4 = '0.4' == torch.__version__[:3]

# --
# Helpers

def _set_train(x, mode):
    x.training = False if getattr(x, 'frozen', False) else mode
    for module in x.children():
        _set_train(module, mode)
    
    return x

# --
# Model

class BaseNet(nn.Module):
    
    def __init__(self, loss_fn=F.cross_entropy, verbose=False):
        super(BaseNet, self).__init__()
        self.loss_fn  = loss_fn
        self.opt      = None
        self.progress = 0
        self.epoch    = 0
        self.lr       = -1
        self.verbose  = verbose
        self._cuda    = False
    
    def cuda(self, device=None):
        self._cuda = True
        super().cuda(device=device)
        return self
    
    def cpu(self):
        self._cuda = False
        super().cpu()
        return self
    
    # --
    # Optimization
    
    def init_optimizer(self, opt, params, lr_scheduler=None, clip_grad_norm=0, **kwargs):
        self.clip_grad_norm = clip_grad_norm
        
        if lr_scheduler is not None:
            assert 'lr' not in kwargs, "BaseWrapper.init_optimizer: can't set LR and lr_scheduler"
            self.lr_scheduler = lr_scheduler
            self.opt = opt(params, lr=self.lr_scheduler(0), **kwargs)
            self.set_progress(0)
        else:
            self.lr_scheduler = None
            self.opt = opt(params, **kwargs)
    
    def set_progress(self, progress):
        self.progress = progress
        self.epoch = np.floor(progress)
        if self.lr_scheduler is not None:
            self.lr = self.lr_scheduler(progress)
            LRSchedule.set_lr(self.opt, self.lr)
    
    # --
    # Training states
    
    def train(self, mode=True):
        """ have to override this function to allow more finegrained control """
        return _set_train(self, mode=mode)
    
    # --
    # Batch steps
    
    def train_batch(self, data, target):
        data, target = Variable(data), Variable(target)
        if self._cuda:
            data, target = data.cuda(), target.cuda()
        
        _ = self.train()
        
        self.opt.zero_grad()
        output = self(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad_norm)
        
        self.opt.step()
        
        return output, float(loss)
    
    def eval_batch(self, data, target):
        if TORCH_VERSION_4:
            data = Variable(data, requires_grad=False)
            target = Variable(target, requires_grad=False)
        else:
            data = Variable(data, volatile=True)
            target = Variable(target, volatile=True)
        
        if self._cuda:
            data, target = data.cuda(), target.cuda()
        
        _ = self.eval()
        
        output = self(data)
        loss = self.loss_fn(output, target)
        
        return output, float(loss)
    
    # --
    # Epoch steps
    
    def train_epoch(self, dataloaders, mode='train', num_batches=np.inf):
        assert self.opt is not None, "BaseWrapper: self.opt is None"
        
        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            gen = enumerate(loader)
            if self.verbose:
                gen = tqdm(gen, total=len(loader), desc='train_epoch:%s' % mode)

            total_loss = 0.0
            
            correct, total, loss_hist = 0, 0, []
            for batch_idx, (data, target) in gen:
                self.set_progress(self.epoch + batch_idx / len(loader))
                
                output, loss = self.train_batch(data, target)

                total += data.shape[0]
                total_loss += loss * data.shape[0]

            
            self.epoch += 1
            return {
                 "loss" : total_loss / total,
            }
        
    def eval_epoch(self, dataloaders, mode='val', num_batches=np.inf):
        
        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            gen = enumerate(loader)
            if self.verbose:
                gen = tqdm(gen, total=len(loader), desc='eval_epoch:%s' % mode)

            total_loss = 0.0

            correct, total, loss_hist = 0, 0, []
            for batch_idx, (data, target) in gen:
                
                output, loss = self.eval_batch(data, target)
                loss_hist.append(loss)
                
                correct += (to_numpy(output.float()).argmax(axis=1) == to_numpy(target)).sum()
                total += data.shape[0]

                total_loss += loss * data.shape[0]
                
                if batch_idx > num_batches:
                    break
                
                if self.verbose:
                    gen.set_postfix(acc=correct / total)
            
            return {
                "acc"  : correct / total,
                "loss": total_loss / total,
            }
    
    def predict(self, dataloaders, mode='val'):
        _ = self.eval()
        
        all_output, all_target = [], []
        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            gen = enumerate(loader)
            if self.verbose:
                gen = tqdm(gen, total=len(loader), desc='eval_epoch:%s' % mode)
            
            for batch_idx, (data, target) in gen:
                if TORCH_VERSION_4:
                    data = Variable(data, requires_grad=False)
                else:
                    data = Variable(data, volatile=True)
                
                if self._cuda:
                    data = data.cuda()
                    
                output = self(data)
                all_output.append(output.data.cpu())
                all_target.append(target)
        
        return torch.cat(all_output), torch.cat(all_target)
    
    def save(self, outpath):
        torch.save(self.state_dict(), outpath)
    
    def load(self, inpath):
        self.load_state_dict(torch.load(inpath))


class BaseWrapper(BaseNet):
    def __init__(self, net=None, **kwargs):
        super(BaseWrapper, self).__init__(**kwargs)
        self.net = net
    
    def forward(self, x):
        return self.net(x)

