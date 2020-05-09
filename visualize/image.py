import numpy as np
from IPython import display
from pylab import *
import os
from common.files import listdir_visible
from PIL import Image

def dynamic_plot(X,Y, figsize=[10,5], max_x=None, min_y=None, max_y=None):
    '''plots dependency between X and Y dynamically: after each call current graph is redrawn
    DEMO:
    X=[]
    Y=[]
    for i in range(10):
        X.append(i)
        Y.append(i**.5)
        plot_dynamic_plot(X,Y,[14,10], max_x=10, max_y=4)  
        time.sleep(0.3)    
    '''
    gcf().set_size_inches(figsize)
    cla()
    plot(X,Y)
    if max_x: 
        plt.gca().set_xlim(right=max_x)
    if min_y: 
        plt.gca().set_ylim(bottom=min_y)
    if max_y: 
        plt.gca().set_ylim(top=max_y)
        
    display.display(gcf())
    display.clear_output(wait=True)  


def img_show(img, title='', figsize=[14,8]):
    '''show PIL image [img], giving a title if [title] is provided'''
    figure();
    plt.gcf().set_size_inches(figsize)
    axis('off')
    imshow(img); xticks([]); yticks([]); 
    if title:
        plt.title(title); 


def show_images(path, K=3, figsize=(20,20)):
    '''Display all images in [path] row by row. In each row [K] images are displayed.
    [figsize] controls size of figure containing all images in one row.'''
    
    figure(figsize=figsize)
    files = listdir_visible(path)
    
    for num,file in enumerate(files,start=1):
        img = Image.open(os.path.join(path,file))
        if img.mode!='RGB':
            img = img.convert('RGB')
        
        subplot(int(ceil(len(files)/K)),K,num)
        show_image(img, file);
