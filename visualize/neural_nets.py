#!/usr/bin/env python
# encoding: utf-8
'''Neural nets (pytorch) visualization tools'''


from pylab import *


def show_params_count(net):
    '''Shows the number of parameters for each layer of neural network net and total number of parameters.'''
    
    total_params_count = 0
    print('LAYER, #PARAMS\n')
    d = net.state_dict()
    for param, w in net.state_dict().items():
        params_count = array(w.shape).prod()
        print('%s: %s'%(param, params_count))
        total_params_count+=params_count   
    print('\n#[TOTAL PARAMS]: %s'%total_params_count)



def show_weights_hist(net,figsize=(16,10),show_zero=True):
    '''Plots histograms of weights for each layer of neural network net.'''
    figure(figsize=figsize)
    d = net.state_dict()
    for i,(param, w) in enumerate(d.items(),1):
        ax=subplot(ceil(len(d)/3),3,i)        
        if w.is_cuda:
            w = w.cpu()        
        w = w.numpy().ravel()
        hist(w, bins=max(10,len(w)//50))
        if show_zero:
            axvline(0,ls='--',color='r')
        title('%s: %s weights'%(param,len(w)))




def get_pytorch_model_name(model):
    '''Return name of pytorch model'''
    if hasattr(model,'module'): # DataParallel model
        return model.module.__class__.__name__
    else: # not parallelized model
        return model.__class__.__name__
