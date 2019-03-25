import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import PCA
from common.feature_transformations import get_lda_directions, get_lda_projections
from common.outliers import get_outlier_sels, get_outlier_sels_within_classes
from common.visualize.colors import COLORS
from pylab import *

def plot_fun(fun,area, args_resolution=300, values_resolution=100):
    '''Plot contour of function fun(x), where x is 2 dimensional list.
    Function is plotted on the area specified by area=[x_min,x_max,y_min,y_max].
    Along each axis <args_resolution> splits are made.
    Along values of the functions <values_resolution> splits are made.
    
    Example: plot_fun(lambda x:(x[0]-1)**2+(x[1]-3)**2,[0,5,0,5])'''
    
    xx1=linspace(area[0], area[1], args_resolution)
    xx2=linspace(area[2], area[3], args_resolution)
    Z=zeros([args_resolution, args_resolution])
    for j,x1 in enumerate(xx1):
        for i,x2 in enumerate(xx2):
            Z[i,j] = fun([x1,x2])
    z=Z.ravel()
    contourf(xx1,xx2,Z,levels=linspace(z.min(),z.max(), values_resolution))
        