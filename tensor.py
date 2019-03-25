#!/usr/bin/env python
# encoding: utf-8

import torch


def flip(x,dim):
    '''Flip tensor along dimension dim.
    Example: 
    A=torch.FloatTensor([[1,2],[3,4]]).cuda()
    A,flip(A,1)'''
    inv_idx = torch.arange(x.shape[dim]-1, -1, -1).long()
    if x.is_cuda:
        dev = x.get_device()
        inv_idx = inv_idx.cuda(dev)
    return torch.index_select(x,dim,inv_idx)



def crop1d(x,cutoff,dim):
    '''Crops tensor x by cutoff elements from the beginning and the end along dimension dim.
    Example:
    x=torch.FloatTensor([1,2,3,4,5,6,7,8]).cuda(1)
    crop1d(x,2,0) '''    
    idx = torch.arange(cutoff, x.shape[dim]-cutoff).long()
    if x.is_cuda:
        dev = x.get_device()
        idx = idx.cuda(dev)
    return torch.index_select(x,dim,idx)


def crop(x,cutoff,dims):
    '''Crops tensor x by cutoff elements from the beginning and the end along each of the dimensions listed in dims
    Example:
    x=torch.FloatTensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]).cuda(1)
    x,crop(x,1,[0,1]) '''
    for dim in dims:
        x = crop1d(x,cutoff,dim)
    return x



