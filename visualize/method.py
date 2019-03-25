#!/usr/bin/env python
# encoding: utf-8

from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import tree
from sklearn.metrics import accuracy_score
from common.iteration import piter
from common.functions import vec, normalize
from pylab import *


def show_performance(iters, train_err_rates, val_err_rates, train_losses, val_losses, figsize=(14, 5), title_str='', verticals=[], verticals2=[]):
    '''Plots a graph of performance measures of converging algorithm. Can plot dynamic graph if called many times.
    '''
    
    from IPython import display
    
    clf()
    gcf().set_size_inches(*figsize)

    ax1 = gcf().add_subplot(111)
    ax1.plot(iters, train_losses, 'g--', label='train loss')
    ax1.plot(iters, val_losses, 'r--', label='val loss')
    xlabel('iterations')
    legend(loc='upper left')
    ylabel('loss')

    ax2 = gcf().add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(iters, train_err_rates, 'g-', lw=3, label='train err.rate')
    ax2.plot(iters, val_err_rates, 'r-', lw=3, label='val err.rate')
    ax2.yaxis.tick_right();
    ax2.yaxis.set_label_position("right")
    ylabel('err.rate')
    
    for vert in verticals:
        axvline(vert, linestyle=':', color='g')

    for vert in verticals2:
        axvline(vert, linestyle=':', color='b')
        
        
    legend(loc='upper right')
    
    title(title_str)
    display.display(gcf())
    display.clear_output(wait=True)    

'''    
#Demo of show_performance:
iters=[]
train_losses = []
val_losses = []
train_err_rates = []
val_err_rates = []
m=0.01

for i in range(3,16):
    iters.append(i)
    train_losses.append( 100+1/(i*i)+m*randn(1)[0] )
    val_losses.append( 100+1/i+m*randn(1)[0] )
    train_err_rates.append( 1/(i*i)+10*m*randn(1)[0] )
    val_err_rates.append( 1/i+10*m*randn(1)[0] )

    show_performance(iters, train_err_rates, val_err_rates, train_losses, val_losses)    
'''


def plot_predictions_2D(model, X_train, Y_train, task, feature_names=None, 
                     train=True, n=50, cmap=None, point_size=15, 
                     offset=0.05, alpha=1):
    '''Plots decision regions for classifier clf trained on design matrix X=[x1,x2] with classes y.
    n is the number of ticks along each direction
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.
    
    model: prediction model, supporting sklearn interface
    X_train: design matrix [n_objects x n_features]
    Y_train: vector of outputs [n_objects]
    task: either "regression" or "classification"
    feature_names: list of feature names [n_features]
    n: how many bins to use along each dimension
    cmap: matplotlib colormap
    point_size: size of points for scatterplot
    offset: margin size around training data distribution
    alpha: visibility of predictions (0=invisible, 1=fully visible)
    '''    
    
    x1, x2 = X_train[:,0],X_train[:,1]
    if train:
        model.fit(X_train, Y_train)
    
    margin1 = offset*(x1.max()-x1.min())
    margin2 = offset*(x2.max()-x2.min())
        
    # create a mesh to plot in
    x1_min, x1_max = x1.min() - margin1, x1.max() + margin1
    x2_min, x2_max = x2.min() - margin2, x2.max() + margin2
    
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, n),
                         np.linspace(x2_min, x2_max, n))

    X_test = hstack( [vec(xx1.ravel()), vec(xx2.ravel())] )    
    Y_test = model.predict(X_test)
    yy=Y_test.reshape(n,n)   
    

    if task=='regression':
        
        vmin = minimum(min(Y_train),min(Y_test))
        vmax = maximum(max(Y_train),max(Y_test))

        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        if cmap==None:
            cmap = cm.hot
        norm = Normalize(vmin=vmin, vmax=vmax) 

        Y_train = cmap(norm(Y_train))
        yy = cmap(norm(yy))

        img = imshow(yy, extent=(x1_min, x1_max, x2_min, x2_max), interpolation='nearest', origin='lower', alpha=alpha)
        scatter(x1, x2, facecolor=Y_train, lw=1, edgecolor='k', s=point_size)    
        
    elif task=='classification':
        from common.visualize.colors import COLORS
        
        classes = unique(Y_train)
        assert len(classes)<=len(COLORS),'Classes count should be <=%s'%len(COLORS)
        
        y2color = lambda y: COLORS[find(y==classes)[0]]
        Z=zeros([n,n,3])
        for i in arange(n):
            for j in arange(n):
                Z[i,j,:]=y2color(yy[i,j])
        
        img = imshow( Z, extent=(x1_min, x1_max, x2_min, x2_max), interpolation='nearest', origin='lower', alpha=alpha)
        scatter(x1, x2, c=[COLORS[find(classes==y)[0]] for y in Y_train], lw=1, edgecolor='k', s=point_size) 
        
    else:
        raise Exception('task should be either "regression" or "classification"!')

    plt.axis([x1_min, x1_max, x2_min, x2_max])
    
    if feature_names!=None:
        xlabel(feature_names[0])
        ylabel(feature_names[1])



def show_param_dependency(m, X_train, Y_train, X_test, Y_test, param_name, param_vals, loss_fun, x_label=None):
    '''score_fun='accuracy',
       Show plot, showing dependency of score_fun (estimated using CV on X_train, Y_train) 
       on parameter param_name taking values in param_vals.
       Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
       
    
    if x_label is None:
        x_label = param_name
        
    losses = zeros(len(param_vals))

    for i, param_val in enumerate(piter(param_vals)):
        m.set_params(**{param_name:param_val})
        m.fit(X_train, Y_train)
        Y_hat = m.predict(X_test)
        if loss_fun=='error_rate':
                losses[i] = mean(Y_hat!=Y_test)
        elif loss_fun=='MAE':
                losses[i] = mean(abs(Y_hat-Y_test))                
        elif loss_fun=='MSE':
                losses[i] = mean((Y_hat-Y_test)**2)
        elif loss_fun=='RMSE':
                losses[i] = sqrt(mean((Y_hat-Y_test)**2))
        else:
            raise ValueError('Unknown loss %s!'%loss_fun)
            
    xlabel(x_label)
    ylabel(loss_fun)
    plot(param_vals, losses)
    print('Min %s = %.4f for %s=%s' % (loss_fun, min(losses), param_name, param_vals[argmin(losses)]) ) 
	

def show_param_dependency_cv(clf, param_name, param_vals, x_label=None, score_fun='accuracy'):
    '''Show plot, showing dependency of score_fun (estimated using CV on X_train, Y_train) 
       on parameter param_name taking values in param_vals.
       Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
       
    
    if x_label is None:
        x_label = param_name
        
    scores = zeros(len(param_vals))

    for val_num, param_val in enumerate(piter(param_vals)):
        clf.set_params(**{param_name:param_val})
        clf_cv = skl.grid_search.GridSearchCV(clf,param_grid={},scoring=score_fun,n_jobs=1, refit=False)
        clf_cv.fit(X_train, Y_train)
        scores[val_num] = clf_cv.best_score_

    xlabel(x_label)
    ylabel(score_fun)
    plot(param_vals, scores)
    print('Top %s = %.4f for %s=%s' % (score_fun, max(scores), param_name, param_vals[argmax(scores)]) )  
    

    


def print_decision_tree(tree, feature_names=None, class_names=None, offset_unit='    '):
    '''Plots textual representation of rules of a decision tree
    tree: scikit-learn representation of tree
    feature_names: list of feature names. They are set to f1,f2,f3,... if not specified
    offset_unit: a string of offset of the conditional block
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    
    if tree.tree_.n_classes[0]==1:
        task = 'regression'
    else:
        task = 'classification'
    
    if feature_names is None:
        features  = ['f%d'%i for i in tree.tree_.feature]
    else:
        features  = [feature_names[i] for i in tree.tree_.feature]  

    def recurse(task, left, right, threshold, features, node, depth=0):
            offset = offset_unit*depth
            if (threshold[node] != -2):
                    print(offset+"if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                    if left[node] != -1:
                            recurse (task, left, right, threshold, features,left[node],depth+1)
                    print(offset+"} else {")
                    if right[node] != -1:
                            recurse (task, left, right, threshold, features,right[node],depth+1)
                    print(offset+"}")
            else:
                    if task=='regression':
                        print(offset+"return %s" % value[node][0,0] )
                    else: # classification
                        y = argmax(value[node][0])
                        if class_names is None:
                            print(offset+"return class%d"%y)
                        else:
                            print(offset+"return " + class_names[y])

    recurse(task, left, right, threshold, features, 0,0)

    
    
    
def visualize_tree(clf,filename):
    '''Writes visualization of tree clf into pdf file filename.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    

    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename)


    

def print_feature_importances(features, importances, min_importance=0):
    '''Prints feature importances in decreasing order. 

    features: iterable of sequence names
    importances: iterable of feature importances
    min_importance (=0): only features with importance>min_importance are shown.

    Author: Victor Kitov (v.v.kitov@yandex.ru).'''    
    
    assert len(features)==len(importances)
    features=array(features)
    importances=array(importances)
    
    features = features[importances>min_importance]
    importances = importances[importances>min_importance]
    
    inds = np.argsort(importances)[::-1]
    features = features[inds]
    importances = importances[inds]
    
    for feature, importance in zip(features, importances):
        print('%35s: %.4f' % (feature, importance))

    
    
def show_feature_importances(features, importances, min_importance=0, figsize=None):
    '''Plot feature importances in decreasing order as a bar plot.
    
    features: iterable of sequence names
    importances: iterable of feature importances
    min_importance (=0): only features with importance>min_importance are shown
    figsize (optional): size of the figure
    
    Author: Victor Kitov (v.v.kitov@yandex.ru).'''
    assert len(features)==len(importances)
    features=array(features)
    importances=array(importances)
    
    features = features[importances>min_importance]
    importances = importances[importances>min_importance]
    
    inds = np.argsort(importances)
    features = features[inds]
    importances = importances[inds]
    
    N=len(features)
    
    if figsize!=None:
        figure(figsize=figsize)
    barh(arange(N),importances, align='center')
    yticks(arange(N), features)
    xlabel('importance')
    title('Feature importances')
    grid(True)    
    
    
def gb_error_metric_plot(est, X_train,y_train, X_test, y_test, ax=None, label='', train_color=[0,1,0],
                  test_color=[1,0,0], alpha=1.0):
    '''Plots sequential error rate for each step of gradient boosted trees.
       Returs handle to axis where the plot was shown.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
       

    fun = lambda y_test,y_pred: 1-accuracy_score(y_test,y_pred)

    n_estimators = est.get_params()['n_estimators']
    err_train = np.zeros(n_estimators)
    err_test = np.zeros(n_estimators)

    for i, (pred_train,pred_test) in enumerate(zip(est.staged_predict(X_train),est.staged_predict(X_test))):
        err_train[i] = fun(y_train, pred_train)
        err_test[i] = fun(y_test, pred_test)

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

    ax.plot(np.arange(n_estimators) + 1, err_train, color=train_color,
             label='Train', linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, err_test, color=test_color,
             label='Test', linewidth=2, alpha=alpha)
    ax.set_ylabel('Error rate')
    ax.set_xlabel('n_estimators')
    ax.set_ylim((0, 1.1*max(err_train.max(),err_test.max())))
    ax.legend()
    title = '{}: min_train={:.04f}, min_test={:.04f}'.format(label,err_train.min(),err_test.min())
    ax.set_title(title)
    return ax
