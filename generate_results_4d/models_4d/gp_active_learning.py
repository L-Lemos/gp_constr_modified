# -*- coding: utf-8 -*-

import time, os, itertools
import pandas as pd
import numpy as np
from scipy.stats import qmc
from sklearn.metrics import max_error, r2_score, mean_squared_error
from generate_results_4d import maximin_dist, constrained_gp_model, run_trnsys_sim_4d, rae_metric
from scipy.optimize import minimize
from rpy2 import robjects
R = robjects.r
import gc

# In[]

def gp_active_learning(city, case, trial, scaler, numcores_input=15, is_parallel=True, unconstrained=False, first_deriv=True, second_deriv=True, error_tol=0.01, count_tol=2, max_train_points=5000):  
  
    def pool_close(pool):
        pool.stop()
        pool.close() 
        pool.join(1)  

    # Start counting run time
    surrogate_start_time = time.time()                              
  
    # Dataframe to store partial progress of GP regression
    gp_stats = pd.DataFrame(columns = ['Num Points', 'RMSE', 'Max Error', 'R2','RAE'])  
    gp_stats_idx = 0
  
    # Get points for RAE
    df2 = pd.read_csv(os.path.join('generate_results_4d','input_data_4d',f"rae_set_{city.lower().replace(' ','_')}.csv")).to_numpy()                                    

    x_rae = df2[:, 0:4].reshape((-1, 4))
    y_rae_actual = df2[:, -1]
   
    # Set of virtual points
    sampler = qmc.LatinHypercube(d=x_rae.shape[1])
    virtual_points = sampler.random(n=5*x_rae.shape[1])
    
    # Transform variables to normalized values
    x_rae = scaler.transform(x_rae)      
    
    # Training points                                                  
    x_train = maximin_dist(None, numcores_input, x_rae.shape[1]) 
    x_train = np.round(x_train,2) # Round, helps finding redundant points in the next steps
    
    # Update list of already simulated points
    x_sample = x_train.copy()
    x_simulated = x_sample.copy()
    
    # Generate inputs for TRNSYS
    x_sample_trnsys = scaler.inverse_transform(x_sample)
    
    col_area = x_sample_trnsys[:,0].tolist()
    storage = x_sample_trnsys[:,1].tolist()
    q_demand = x_sample_trnsys[:,2].tolist()
    t_demand = x_sample_trnsys[:,3].tolist()

    print('Running TRNSYS')
    start_time = time.time()
    try:
        solfrac = run_trnsys_sim_4d(city , col_area , storage , q_demand, t_demand)
        solfrac = np.array(solfrac).reshape(x_sample.shape[0],1).flatten()
    except:
        df = pd.DataFrame(columns=['Area','Storage','Q_demand','T_demand'])
        df.Area = col_area
        df.Storage = storage
        df.Q_demand = q_demand
        df.T_demand = t_demand
        # df.to_excel(f'Error_inputs_{city}_{trial}.xlsx')
        raise ValueError('ERROR IN TRNSYS SIMULATION')
    
    x_sample = x_sample[~np.isnan(solfrac),:]
    solfrac = solfrac[~np.isnan(solfrac)]
    
    y_train = solfrac
    print(f'{time.time() - start_time:.2f} for TRNSYS simulation')
    
    # Create model
    model = constrained_gp_model(x_train.shape[1],unconstrained,first_deriv,second_deriv)
    
    # Train
    model.train_model(x_train,y_train,virtual_points)
  
    # Get predictions for RAE calculation
    y_rae, _ = model.predict(x_rae)
    
    rmse_rae = np.sqrt(mean_squared_error(y_rae_actual,y_rae))
    r2_rae = r2_score(y_rae_actual,y_rae)
    max_err_rae = max_error(y_rae_actual,y_rae)
    
    gp_stats.loc[gp_stats_idx,:] = [x_train.shape[0], rmse_rae, max_err_rae, r2_rae, np.nan]
    gp_stats_idx += 1
    
    # Counter to check if process has stabilized for enough iterations
    counter = 0
   
    # Counter of iterations
    run_counter = 1                                
    
    # Next iteration does not use maximin
    use_maximin = False
    
    # Add a new training point at each iteration, until the training set is equal to the original set
    while True:
    
        # break    
    
        gc.collect() # Garbage collection
        R('rm(list = ls(all.names=TRUE))') # Refresh R to prevent memory leaks
        R('gc()')# R garbage collection         
      
        if not(use_maximin):
                
            try:
                # Find next point in x-axis to sample, using a "acquisition function"              
                x0 = sampler.random(n=3*numcores_input)
                
                # Improve search, since acq_selected was taken from a coarse grid
                def func(x):
                    x = x.reshape(1,-1)
                    m, v = model.predict(x)
                    return -v
                
                opt_bounds = list()
                for dim in range(0,x_train.shape[1]):
                    opt_bounds.append((0,1))
                    
                print('Finding max variance points')
                start_time = time.time()              
                    
                x_sample = list()
                for x in x0:
                    res = minimize(func,x,bounds=opt_bounds,method='Nelder-Mead')
                    x_sample.append(res.x)
                x_sample = np.array(x_sample)                                            
                
                print(f'{time.time() - start_time:.2f} to find maximum variance locations')
                
                x_sample = np.round(x_sample,2) # Round to two decimals, to help eliminating points too close to each other                                                        
                
                x_sample = np.unique(x_sample,axis=0)
                
                # New samples cannot contain points too close to already simulated points 
                dist = [min([np.linalg.norm(y - x) for x in x_simulated]) for y in x_sample]
                x_sample_idx = [x > 0.05 for x in dist]
                x_sample = x_sample[x_sample_idx,:]
                
                # Sort remaining points based on predictive variance
                _, var_val = model.predict(x_sample)
                idx = np.flip(np.argsort(var_val))
                var_val = var_val[idx].squeeze()
                x_sample = x_sample[idx,:].squeeze()
                x_sample = x_sample.reshape(-1,x_train.shape[1])
                
                # Exclude sample points that are too close to each other                                                        
                def remove_too_close(input_list, threshold=0.05):
                    
                    combos = itertools.combinations(input_list, 2)
                    points_to_remove = np.array([point2 for point1, point2 in combos])
                    combos = itertools.combinations(input_list, 2)
                    dists = np.array([np.linalg.norm(point1 - point2) for point1, point2 in combos])
                    points_to_remove = points_to_remove[dists <= threshold]
                    points_to_keep = np.array([point for point in input_list if (point.tolist() not in points_to_remove.tolist())])
                    return points_to_keep                                                        
                
                x_sample = remove_too_close(x_sample) 
            
            except:
                
                # If it did not work (could not calculate variance), use maximin
                use_maximin = not(use_maximin)
                continue
                
        else:
        
            x_sample = maximin_dist(x_simulated, numcores_input, x_simulated.shape[1]) 
            
        
        if x_sample.shape[0] >= numcores_input: # If there are too many points to sample from, pick only some
            
            x_sample = x_sample[0:numcores_input,:]            
        
        else: # If there are too few, use maximin distance
            
            try:
                # Get samples    
                new_samples = maximin_dist(np.concatenate((x_simulated,x_sample.reshape(-1,4)),axis=0), numcores_input-x_sample.shape[0], x_train.shape[1])
            
                # Concatenate
                x_sample = np.concatenate((x_sample,new_samples),axis=0)  
                
            except:
                
                # Get samples    
                x_sample = maximin_dist(x_simulated, numcores_input, x_train.shape[1])
                                          
        
        # Update list of already simulated points
        x_simulated = np.concatenate([x_simulated,x_sample],axis=0)

        # Generate inputs for TRNSYS
        x_sample_trnsys = scaler.inverse_transform(x_sample)

        col_area = x_sample_trnsys[:,0].tolist()
        storage = x_sample_trnsys[:,1].tolist()
        q_demand = x_sample_trnsys[:,2].tolist()
        t_demand = x_sample_trnsys[:,3].tolist()

        print('Running TRNSYS')
        start_time = time.time()
        try:
            solfrac = run_trnsys_sim_4d(city , col_area , storage , q_demand, t_demand)
            solfrac = np.array(solfrac).reshape(numcores_input,1).flatten()
        except:           
            df = pd.DataFrame(columns=['Area','Storage','Q_demand','T_demand'])
            df.Area = col_area
            df.Storage = storage
            df.Q_demand = q_demand
            df.T_demand = t_demand
            # df.to_excel(f'Error_inputs_{city}_{trial}.xlsx')
            raise ValueError('ERROR IN TRNSYS SIMULATION')
        print(f'{time.time() - start_time:.2f} for TRNSYS simulation')                                           

        x_sample = x_sample[~np.isnan(solfrac),:]
        solfrac = solfrac[~np.isnan(solfrac)]
        
        # Add to the training set
        x_train = np.append(x_train, x_sample, axis=0)
        y_train = np.append(y_train, solfrac, axis=0)
            
        y_rae_old = y_rae.copy()

        x_train = np.round(x_train,2) # Round, helps finding redundant points in the next steps

        # Create model again
        model = constrained_gp_model(x_train.shape[1],unconstrained,first_deriv,second_deriv)  
    
        # Train
        model.train_model(x_train,y_train,virtual_points)      

        y_rae, _ = model.predict(x_rae)
    
        rae = rae_metric(y_rae, y_rae_old)
        error_val = rae
        
        rmse_rae = np.sqrt(mean_squared_error(y_rae_actual,y_rae))
        r2_rae = r2_score(y_rae_actual,y_rae) 
        max_err_rae = max_error(y_rae_actual,y_rae) 
        
        gp_stats.loc[gp_stats_idx,:] = [x_train.shape[0], rmse_rae, max_err_rae, r2_rae, error_val]
        gp_stats_idx += 1

        # Change the sampling method for the next iteration
        use_maximin = not(use_maximin) 

        print(f'RAE = {error_val:.2f}')

        # gp_stats.to_excel(os.path.join('results',f'stats_for_case_{case}_{trial}.xlsx'))

        # Check convergence
        if error_val < error_tol:
            counter += 1  # Update counter if error is below tolerance
        else:
            counter = 0  # Else, reset counter
    
        if counter == count_tol:
            break  # If there is no change in regression model for enough iterations, break
            
        run_counter += 1
        
        if x_train.shape[0] >= max_train_points: 
            break
    
    # pool_close(pool)
    
    total_time = time.time() - surrogate_start_time
    
    return model, x_train, max_err_rae, rmse_rae, r2_rae, total_time