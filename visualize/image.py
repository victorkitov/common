import numpy as np
from pylab import plt
from IPython import display

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
