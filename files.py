#!/usr/bin/env python
# encoding: utf-8

import os, shutil


def listdir_visible(path):
    '''list all visible files in directory'''
    return [f for f in os.listdir(path) if not f.startswith('.')]

def recreate_dir(path):
    '''If directory in path doesn't exist, it will be created. If it already exists, it will be deleted and recreated. May be used to output fresh experiment results.'''
    if os.path.exists(path):    
        shutil.rmtree(path)    # remove previous results (if any)
    os.makedirs(path)     
