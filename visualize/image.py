import numpy as np
from IPython import display
from pylab import *
import os
from common.files import listdir_visible
from PIL import Image

def dynamic_img_show(img,title_str='',fig_size=[14,8],hide_axes=True):
    '''Show image <img>. If called repeatedly within a cycle will dynamically redraw image.
    #DEMO
    import time

    for i in range(10):
        img = np.zeros([50,50])
        img[:i*5]=1
        dynamic_img_show(img,'iter=%s'%i)
        time.sleep(0.1)
    '''
    plt.clf()
    plt.title(title_str)
    plt.imshow(img)
    plt.xticks([]); plt.yticks([]);
    plt.gcf().set_size_inches(fig_size)
    display.display(plt.gcf())
    display.clear_output(wait=True)   


def show_image(pil_image, title_str):
    '''show PIL image [pil_image], giving a title if [title_str] is provided'''
    imshow(pil_image); xticks([]); yticks([]); 
    if title_str:
        title(title_str); 


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
