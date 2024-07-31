# -*- coding: utf-8 -*-
"""
This function replaces the original MATLAB + TRNSYS simulation
An RBF interpolant is used, built using 300 previously simulated data points
A time.sleep() can be inserted to emulate the time required for running TRNSYS with 15 parallel simulation cores
"""

import pandas as pd
import numpy as np
import os
from scipy.interpolate import RBFInterpolator
import time

# In[]

def run_trnsys_sim_2d(col_area , storage):

    # Reference data for comparing generated models
    df2 = pd.read_csv(os.path.join('simplified_example_2d','input_data','trnsys_emulation_set_iguatu_2d.csv')).to_numpy()     
    x_train = df2[:, 0:2].reshape((-1, 2))
    y_train = df2[:, -1].flatten()
    
    # Adjust interpolant
    model = RBFInterpolator(x_train, y_train)
    
    # Organize inputs to query
    x_query = np.array([col_area, storage]).transpose()
    
    # Run query on interpolant
    solfrac = model(x_query)
    solfrac[solfrac < 0] = 0
    solfrac[np.isnan(solfrac)] = 0
    
    # time.sleep(130) # Emulation of TRNSYS simulation time with 15 cores parallel computing
    
    return solfrac