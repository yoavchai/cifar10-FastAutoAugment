#!/usr/bin/env python

"""
    lr.py
    
    learning rate scheduler
"""

from __future__ import print_function, division

import sys
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

# --
# Helpers

def power_sum(base, k):
    return (base ** (k + 1) - 1) / (base - 1)


def inv_power_sum(x, base):
    return np.log(x * (base - 1) + 1) / np.log(base) - 1

# --

class LRSchedule(object):
    
    @staticmethod
    def set_lr(optimizer, lr):
        num_param_groups = len(list(optimizer.param_groups))
        
        if isinstance(lr, float):
            lr = [lr] * num_param_groups
        else:
            assert len(lr) == num_param_groups, "len(lr) != num_param_groups"
        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr[i]
    
    @staticmethod
    def constant(lr_init=0.1, **kwargs):
        def f(progress):
            return lr_init
        
        return f
    
    @staticmethod
    def step(lr_init=0.1, breaks=(150, 250), factors=(0.1, 0.1), **kwargs):
        """ Step function learning rate annealing """
        assert len(breaks) == len(factors)
        breaks = np.array(breaks)
        def f(progress):
            return lr_init * np.prod(factors[:((progress >= breaks).sum())])
        
        return f


    @staticmethod
    def linear_cycle(lr_init=0.1, epochs=10, low_lr=0.005, extra=5, **kwargs):
        def f(progress):
            if progress < 15:  # TODO was changed by yoav
                # if progress < epochs / 2:
                return 2 * lr_init * (1 - float(epochs - progress) / epochs)
            elif progress <= epochs:
                return low_lr + 2 / 6 * lr_init * float(epochs - progress) / epochs
            elif progress <= epochs + extra:
                return low_lr * float(extra - (progress - epochs)) / extra
            else:
                return low_lr / 10

        return f

    

    




