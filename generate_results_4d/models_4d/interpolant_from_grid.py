# -*- coding: utf-8 -*-


import time, os
import pandas as pd
import numpy as np
from sklearn.metrics import max_error, r2_score, mean_squared_error
from generate_results_4d import  run_trnsys_sim_4d
import matplotlib.pyplot as plt
from rpy2 import robjects
R = robjects.r

# In[]

def interpolant_from_grid(city, interpolant, x_train, scaler, numcores_input=4, is_parallel=True):
    
    #Start counting run time
    surrogate_start_time = time.time()
    
    # Reference data for comparing generated models
    df2 = pd.read_csv(os.path.join('generate_results_4d','input_data_4d',f"rae_set_{city.lower().replace(' ','_')}.csv")).to_numpy()                                    
    
    x_ref = df2[:, 0:4].reshape((-1, 4))
    y_ref_actual = df2[:, -1]
    
    x_ref = scaler.transform(x_ref)
    
    sample_size = x_train.shape[0]
    
    y_train = np.empty(0)
    
    print('Running TRNSYS on selected grid of points')
    
    for ii in range(0,int(np.ceil(sample_size/numcores_input))):  
    
        start = numcores_input*ii
        end = min(numcores_input*(ii+1), sample_size)
   
        x_sample_trnsys = scaler.inverse_transform(x_train[start:end,:]) 
   
        col_area = x_sample_trnsys[:,0].tolist()
        storage = x_sample_trnsys[:,1].tolist()
        q_demand = x_sample_trnsys[:,2].tolist()
        t_demand = x_sample_trnsys[:,3].tolist()

        solfrac = run_trnsys_sim_4d(city , col_area , storage , q_demand, t_demand)
        solfrac = np.array(solfrac)
        try:
            solfrac = solfrac.reshape(solfrac.shape[0],1).flatten()
        except:
            solfrac = solfrac.flatten()        
            
        y_train = np.concatenate((y_train,solfrac),axis=0)        
    
    x_data_full = x_train.copy()
    y_data_full = y_train.copy()
    
    fig,ax = plt.subplots(1)  
    
    for asf in np.unique(x_data_full[:,0]):
        for rst in np.unique(x_data_full[:,1]):
            for td in np.unique(x_data_full[:,3]):
                
                idx = x_data_full[:,0] == asf
                idx = idx & (x_data_full[:,1] == rst)
                idx = idx & (x_data_full[:,3] == td)                        
                                   
                ax.plot(x_data_full[idx,2], y_data_full[idx])
                
    ax.set_ylabel('Solar Fraction')
    ax.set_xlabel('Demanded heat')
    plt.savefig(os.path.join('generate_results_4d','output_data_4d',f'Heat partial influence {city}.png'),bbox_inches='tight',dpi=300)
                
    fig,ax = plt.subplots(1)  
    
    for asf in np.unique(x_data_full[:,0]):
        for rst in np.unique(x_data_full[:,1]):
            for qd in np.unique(x_data_full[:,2]):
                
                idx = x_data_full[:,0] == asf
                idx = idx & (x_data_full[:,1] == rst)
                idx = idx & (x_data_full[:,2] == qd)                        
                                   
                ax.plot(x_data_full[idx,3], y_data_full[idx])
                
    ax.set_ylabel('Solar Fraction')
    ax.set_xlabel('Demand temperature')
    plt.savefig(os.path.join('generate_results_4d','output_data_4d',f'Temperature partial influence {city}.png'),bbox_inches='tight',dpi=300)
    
    idx = ~np.isnan(y_train)
    x_train = x_train[idx,:]
    y_train = y_train[idx]
    
    model = interpolant(x_train, y_train.flatten())
                                            
    # Get model prediction errors
    y_ref = model(x_ref)                                                 
    
    # Check for NaN in interpolation results (The LinearNDInterpolator does not
    # work well when x_rae is on the edges of the input domain)
    nan_idx = np.isnan(y_ref)
    if nan_idx.sum() > 0:
        y_ref = y_ref[~nan_idx]
        y_ref_actual = y_ref_actual[~nan_idx]
    
    rmse_rae = np.sqrt(mean_squared_error(y_ref_actual,y_ref))
    r2_rae = r2_score(y_ref_actual,y_ref) 
    max_err_rae = max_error(y_ref_actual,y_ref)
    
    total_time = time.time() - surrogate_start_time
    
    pd.DataFrame(np.concatenate((x_train,y_train.reshape(-1,1)),axis=1)).to_csv(os.path.join('generate_results_4d','output_data_4d',f'Simulated_cases_{city}.csv'),index=False)
    
    return model, x_train, max_err_rae, rmse_rae, r2_rae, total_time, nan_idx