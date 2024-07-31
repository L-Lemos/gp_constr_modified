# -*- coding: utf-8 -*-

import numpy as np

# RAE metric used to monitor surrogate model maturity
def rae_metric(y_new, y_old):
    
    return ((np.abs(y_old - y_new))/(np.abs(y_new).max())).sum()/y_new.shape[0]