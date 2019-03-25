from pylab import *
from sklearn.decomposition import PCA
from common.visualize.colors import COLORS


def pca_2D(X, Y=None, task=None, cm=None, point_size=10, figsize=None):
    '''Display data in the space of first two principal components.
    Data is internally standardized to zero mean and unit variance. Constant features are removed.
    
    X: matrix [n_objects x n_features]
    Y: vector of outputs [n_objects]
    task: None or 'classification' or 'regression'
    cm: matplotlib colormap for color display for regression
    '''
    
    
    D=X.shape[1]
    pca = PCA(n_components=2)
    
    const_features_sels = (X.std(0)==0)
    X=X[:,~const_features_sels]  # remove constant features
    
    X = (X-X.mean(0))/X.std(0)  # data standardization
    
    X_pca = pca.fit_transform(X)
    cum_explained_variance = cumsum(pca.explained_variance_ratio_)
	
    if figsize:
        figure(figsize=figsize)
        
    if (task == None):
        assert Y==None,'<Y> should not be specified if <task> is not specified.'
        scatter(X_pca[:,0], X_pca[:,1])
    else:
        if task=='classification':
            scatter(X_pca[:,0], X_pca[:,1], c=[COLORS[y] for y in Y], s=point_size)
        elif task=='regression':
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
			
            if cmap==None:
                cmap = cm.ocean
            norm = Normalize(vmin=min(Y), vmax=max(Y)) 		
            scatter(X_pca[:,0], X_pca[:,1], c=[cmap(norm(y)) for y in Y], s=point_size)
        else:
            raise Exception('<task> should be either "regression" or "classification"!')
			
    xlabel('principal component 1')
    ylabel('principal component 2')
    title('First 2 components explain %.3f variance'%cum_explained_variance[1])
        
        

def discr_dist_classification(X,Y, feature_name='', p_min=None, p_max=None, normed=True):
    
    if p_min:
        x = percentile(X,p_min)
        sels = (X>=x)
        X=X[sels]
        Y=Y[sels]
    if p_max:
        x = percentile(X,p_max)
        sels = (X<=x)
        X=X[sels]
        Y=Y[sels]
        
    classes = unique(Y)
    X_unique=unique(X)
    K = len(X_unique)
    ind = np.arange(K)    # the x locations for the groups

    width = 0.35       # the width of the bars: can also be len(x) sequence

    bottom=zeros(K)
    x_counts=zeros(K)
    pp=[]
    for y_num, y in enumerate(classes):
        class_counts=zeros(len(X_unique))
        for x_num,x in enumerate(X_unique):
            if normed:
                x_counts[x_num] = sum((X==x) & (Y==y)) / sum( (X==x) )
            else:
                x_counts[x_num] = sum((X==x) & (Y==y))
        p = plt.bar(ind, x_counts, width, bottom=bottom, fc=COLORS[y_num])
        bottom+=x_counts
        pp.append(p)

    plt.ylabel('Classes counts')
    plt.xticks(ind, X_unique)

    if feature_name:
            title('Distribution of %s'%feature_name)
    plt.legend([p[0] for p in pp], classes, loc='best')
    
    
    

def cont_dist_classification(X,Y, feature_name='', p_min=None, p_max=None, normed=True, bins=100):
    
    classes = unique(Y)
    if p_min:
        x = percentile(X,p_min)
        sels = (X>=x)
        X=X[sels]
        Y=Y[sels]
    if p_max:
        x = percentile(X,p_max)
        sels = (X<=x)
        X=X[sels]
        Y=Y[sels]
    
    if normed==False:
        hist([X[Y==y] for y in classes], stacked=True, color=[COLORS[i] for i in range(len(classes))], 
             bins=bins)
    else: # normed=True     
        bottom=zeros(bins)
        for y_num,y in enumerate(classes):
            a,b,c = hist(X[Y==y], normed=True, color=COLORS[y_num], 
                 bins=linspace(min(X),max(X),bins+1), bottom=bottom)
            bottom+=a
#             #hist(X[Y==y], normed=True, color=COLORS[y_num], 
#             #     bins=bins, alpha=0.5)

    import matplotlib.patches as mpatches
    recs = []
    for y in classes:
        recs.append(mpatches.Rectangle((0,0),1,1,fc=COLORS[find(classes==y)[0]]))
    plt.legend(recs,classes,loc='best')
    if feature_name:
        title('Distribution of %s'%feature_name)
        
        
        

def cross_distributions(X, feature_names=None, bins=30, point_size=5, figsize=None):
    """Generate scatter plots of all pairs of variables. Variables are columns of matrix X.
    
    Input:
        X: matrix n_objects x n_features
        feature_names: list of feature names
        bins: number of bins for each histogram on the diagonal
        point_size: size of each point on the scatterplot
        figsize: tuple (X_size, Y_size)
    """

    nVariables = X.shape[1]
    assert nVariables<50, 'nVariables should be less than 50.'
    
    if feature_names is None:
        feature_names = ['x%d'%i for i in range(nVariables)]
    else:
        assert len(feature_names)==nVariables,'len(feature_names) should be equal to variables count.'

    if figsize==None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    
    for i in range(nVariables):
        for j in range(nVariables):
            nSub = i * nVariables + j + 1
            ax = fig.add_subplot(nVariables, nVariables, nSub)
            
            if i == 0:  # horizontal variable names
                ax.set_xlabel(feature_names[j]) 
                ax.xaxis.set_label_position('top')

            if j == 0:  # vertical variable names
                ax.set_ylabel(feature_names[i])            
            
            if i == j:
                ax.hist(X[:,i],bins=bins)
            else:
                ax.scatter(X[:,j], X[:,i], c='b', lw=0, s=point_size)
            ax.set_xticks([])
            ax.set_yticks([])
			
			
			

def cross_distributions_classification(X, Y, var_names=None, bins=30, point_size=5, figsize=None):
    """Generate scatter plots of all pairs of features, coloring objects with their class. 
    Features are columns of matrix X. Objects are rows. Y stores classes of objects.
    
    Input:
        X: matrix n_objects x n_features
        Y: stores classes of objects.
        feature_names: list of feature names
        bins: number of bins for each histogram on the diagonal
        point_size: size of each point on the scatterplot        
        figsize: tuple (X_size, Y_size)
    """
    
    classes = unique(Y)
    assert len(classes)<=len(COLORS),'Classes count should be <=%s'%len(COLORS)
    
    nVariables = X.shape[1]
    assert nVariables<40, 'nVariables should be less than 40.'
    
    if var_names is None:
        var_names = ['x%d'%i for i in range(nVariables)]
    else:
        assert len(var_names)==nVariables,'len(var_names) should be equal to variables count.'

    if figsize==None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    
    for i in range(nVariables):
        for j in range(nVariables):
            nSub = i * nVariables + j + 1
            ax = fig.add_subplot(nVariables+1, nVariables, nSub)
            
            if i == 0:  # horizontal variable names
                ax.set_xlabel(var_names[j]) 
                ax.xaxis.set_label_position('top')

            if j == 0:  # horizontal variable names
                ax.set_ylabel(var_names[i])            
            
            if i == j:
                ax.hist([X[Y==y,i] for y in classes], bins=bins, stacked=True, color=[COLORS[i] for i in range(len(classes))])
                #ax.hist(X[:,i],bins=bins)
            else:
                ax.scatter(X[:,j], X[:,i], c=[COLORS[find(classes==y)[0]] for y in Y], lw=0, s=point_size)
            ax.set_xticks([])
            ax.set_yticks([])
    
    ax = fig.add_subplot(nVariables+1, nVariables, nSub+1)
    import matplotlib.patches as mpatches
    recs = []
    for y in classes:
        recs.append(mpatches.Rectangle((0,0),1,1,fc=COLORS[find(classes==y)[0]]))
    plt.legend(recs,classes,loc='center')
    axis('off');

			
			
def cross_distributions_regression(X, Y, var_names=None, bins=30, point_size=5, cmap=None, figsize=None):
    """Generate scatter plots of all pairs of features, coloring objects with their output regression value. 
    Features are columns of matrix X. Objects are rows of X. Y stores output values for each object.
    
    Input:
        X: matrix n_objects x n_features
        Y: stores classes of objects.
        feature_names: list of feature names
        bins: number of bins for each histogram on the diagonal
        point_size: size of each point on the scatterplot        
        figsize: tuple (X_size, Y_size)
    """

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    if cmap==None:
        cmap = cm.hot #cm.ocean
    norm = Normalize(vmin=min(Y), vmax=max(Y)) 
    
    nVariables = X.shape[1]
    assert nVariables<50, 'nVariables should be less than 50.'
    
    if var_names is None:
        var_names = ['x%d'%i for i in range(nVariables)]
    else:
        assert len(var_names)==nVariables,'len(var_names) should be equal to variables count.'

    if figsize==None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    
    for i in range(nVariables):
        for j in range(nVariables):
            nSub = i * nVariables + j + 1
            ax = fig.add_subplot(nVariables, nVariables, nSub)
            
            if i == 0:  # horizontal variable names
                ax.set_xlabel(var_names[j]) 
                ax.xaxis.set_label_position('top')

            if j == 0:  # horizontal variable names
                ax.set_ylabel(var_names[i])            
            
            if i == j:
                ax.hist(X[:,i],bins=bins)
            else:
                ax.scatter(X[:,j], X[:,i], c=[cmap(norm(y)) for y in Y],lw=0,s=point_size)
            ax.set_xticks([])
            ax.set_yticks([])