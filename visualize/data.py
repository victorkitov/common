import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import PCA
from common_plus.outliers import get_outlier_sels, get_outlier_sels_within_classes
from common.visualize.colors import COLORS
from pylab import *


def pca_explained(X):
    '''Shows how much variance is explained by each number of principal components.
    Features are automatically standardized to zero mean and unit variance. Constant features are removed.
    
    X: matrix [n_objects x n_features]'''
    
    const_features_sels = (X.std(0)==0)
    X=X[:,~const_features_sels]  # remove constant features
    
    X = (X-X.mean(0))/X.std(0)  # data standardization
    
    D=X.shape[1]
    pca = PCA()
    X_pca = pca.fit_transform(X)
    cum_explained_variance = cumsum(pca.explained_variance_ratio_)
	    
    figure()
    plot(range(1,D+1), cum_explained_variance)
    xticks( range(1,D+1) )
    xlabel('# components')
    ylabel('explained variance fraction')
    for threshold in [0.9, 0.95, 0.99]:
        ind = find(cum_explained_variance>threshold)[0]
        print('%.2f of variance is explained by %d components'% (threshold,ind+1))
    show()
        
        
        
    

        

def plot_corr(df,size=10,show_colorbar=True,show_grid=True):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
        
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
        

    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.1),
                       (1.0, 1.0, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue':  ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
            }

    from matplotlib.colors import LinearSegmentedColormap
    blue_red_cmap = LinearSegmentedColormap('hot', cdict)


    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    plt.set_cmap(blue_red_cmap)
    m = ax.matshow(corr,interpolation='none', vmin=-1,vmax=1) #
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    if show_colorbar is True:
        plt.colorbar(m)
    if show_grid is True:
        plt.grid(color=[0.5,0.5,0.5], linestyle=':', linewidth=1)
