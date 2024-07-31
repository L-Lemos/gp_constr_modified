
### Python wrappers for R functions ###
import rpy2.robjects as robjects
import numpy as np
import os 

# Set source
r_source = robjects.r['source']

# Set working directory for R   
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\','/')
robjects.r("setwd('{}')".format(dir_path))

# Import custom files and define functions using R
r_source("maximin_func.R")
r_maximin = robjects.r['maxmin_dist']

def param_py_to_r(Xorig):
    """ Convert to r objects """
   
    Xorigflat = Xorig.flatten().tolist()
    r_Xorig = robjects.r['matrix'](robjects.FloatVector(Xorigflat), nrow = Xorig.shape[0], byrow=True)
    
    return r_Xorig
    
def maximin_dist_sample(n, p, T, Xorig, verb=False, boundary=False):
    
    robjects.r('rm(list = ls(all.names=TRUE))') # Refresh R to prevent memory leaks
    robjects.r('gc()')# R garbage collection  
    
    # Convert to R objects
    if Xorig is not None:
        r_Xorig = param_py_to_r(Xorig)
    else:
        r_Xorig = robjects.NULL
    
    # Run R function and cast to numpy array
    while True: # Have to run it on a while loop due to a seemingly random "DD2 object not found" error
        try:
            X = r_maximin(n, p, T, r_Xorig, verb, boundary)
            break
        except:
            pass
    X = np.array(X[0])
    
    assert not np.isnan(np.min(X)), 'maximin returns nan'
    
    return X