# -*- coding: utf-8 -*-

import numpy as np
from .maximin_python_wrappers import maximin_dist_sample
from scipy.spatial.distance import pdist, squareform
import time
from scipy.stats import qmc

# In[]

def maximin_dist(x0, n, dims): 
    
    print('Sampling with maximin distance criterion')
    sstart = time.time()
    
    if n > dims: # If number of points is larger than dimension of space to sample from, use Sun's algorithm
    
        x = maximin_dist_sample(n, dims, 100*n*dims, x0, False, False)
        
        if x0 is not None:
            x = x[x0.shape[0]:,:]
    
    else: # If not, generate dims + 1 samples and then choose n
        
        x = maximin_dist_sample(dims + 1, dims, 100*n*dims, x0, False, False)      
            
        # Selected the points with larger average distance from the others
        dists = squareform(pdist(x))
        dists[dists==0] = np.nan
        dists = np.nanmean(dists,axis=0)
        dists = dists[x0.shape[0]:]
        
        if x0 is not None:
            x = x[x0.shape[0]:,:]
        
        idx = np.flip(np.argsort(dists))
        x = x[idx,:]
        x = x[0:n,:]      
    
    print(f'{time.time() - sstart:.2f} seconds for maximin sampling')
    
    return x