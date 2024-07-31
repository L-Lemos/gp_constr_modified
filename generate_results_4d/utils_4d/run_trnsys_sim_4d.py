# -*- coding: utf-8 -*-
"""
This function replaces the original MATLAB + TRNSYS simulation
An RBF interpolant is used, built using 4000 previously simulated data points
A time.sleep() is inserted to emulate the time required for running TRNSYS with 15 parallel simulation cores
"""

import pandas as pd
import numpy as np
import os
from scipy.interpolate import RBFInterpolator
import time

# In[]

def run_trnsys_sim_4d(city , col_area , storage , q_demand, t_demand):

    # Reference data for comparing generated models
    df2 = pd.read_csv(os.path.join('generate_results_4d','input_data_4d',f"testing_set_{city.lower().replace(' ','_')}.csv")).to_numpy()     
    x_train = df2[:, 0:4].reshape((-1, 4))
    y_train = df2[:, -1].flatten()
    
    # Adjust interpolant
    model = RBFInterpolator(x_train, y_train)
    
    # Organize inputs to query
    x_query = np.array([col_area, storage, q_demand, t_demand]).transpose()
    
    # Run query on interpolant
    solfrac = model(x_query)
    solfrac[solfrac < 0] = 0
    solfrac[np.isnan(solfrac)] = 0
    
    time.sleep(375) # Emulation of TRNSYS simulation time with 15 cores parallel computing
    
    return solfrac