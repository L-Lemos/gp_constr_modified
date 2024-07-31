# -*- coding: utf-8 -*-

# Import necessary libraries, point to R home folder
import os
os.environ['R_HOME'] = os.path.join('C:',os.sep,'R-4.3.2')

import numpy as np
import pandas as pd
from sklearn.metrics import max_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RBFInterpolator
from generate_results_4d import gp_active_learning, interpolant_from_grid, generate_violin_plots

# Create R instance
from rpy2 import robjects
R = robjects.r

# In[]

if __name__ == '__main__':
    
    # Turn to true if the script is to generate the plots from scratch
    # If False, script will keep from where last run has stopped
    # This is useful since the figure will take a long time to generate
    # Results of each run are stored in output_data_4d/stats_table_iguatu
    restart = False
  
    # RAE below which active learning is to stop iterating
    error_tol = 0.01
    
    # Active learning stops if RAE is below error_tol for count_tol iterations
    count_tol = 4
    
    # Number of simulation cores for TRNSYS to use
    numcores_input = 15
    
    # Number of active learning runs used to build violin plots
    num_trials = 50
    
    # Iterate through cities (in this example, only Iguaty is available...)
    for city in ['Iguatu']:

        # Import grid to simulate
        x_data_full = pd.read_csv(os.path.join('generate_results_4d','input_data_4d','gridded_x_input.csv')).to_numpy()
                
        # Normalize
        domain_limits = [x_data_full.min(axis=0).tolist(),x_data_full.max(axis=0).tolist()]
        scaler = MinMaxScaler().fit(domain_limits)
        x_data_full = scaler.transform(x_data_full)         

        # Reference data for assessing accuracy generated models
        df2 = pd.read_csv(os.path.join('generate_results_4d','input_data_4d',f"testing_set_{city.lower().replace(' ','_')}.csv")).to_numpy()  
        x_ref = df2[:, 0:4].reshape((-1, 4))
        y_ref_actual = df2[:, -1]

        # File to save results of models performance
        filename = os.path.join('generate_results_4d','output_data_4d',f"stats_table_{city.lower().replace(' ','_')}.xlsx")
        
        # Create dataframe to get surrogates statistics
        if restart:
            stats_table = pd.DataFrame(columns = ['Model','Num Points', 'Total Time', 'RMSE', 'Max Error', 'R2', 'RMSE Full', 'Max Error Full', 'R2 Full'])           
        else:    
            try:
                stats_table = pd.read_excel(filename, index_col=0)                
            except:
                stats_table = pd.DataFrame(columns = ['Model','Num Points', 'Total Time', 'RMSE', 'Max Error', 'R2', 'RMSE Full', 'Max Error Full', 'R2 Full'])               
        
        ii = 0 # Start counter of runs
        grid_ok = False # Flag indicating whether the grid-based interpolant has already been generated
        
        for trial in range(0,num_trials): # Go from trial to trial of active learning process
        
            for case in list(range(1,5)): # Go through surrogate types
            
                # If grid-based interpolant has already been generated, continue
                if case==4 & grid_ok: 
                    continue
            
                # If this run has already been executed and its results stored, continue
                if ii in stats_table.index: 
                    ii += 1
                    continue
                
                # Make sure not too many runs are made
                if ii >= 3*num_trials + 1:
                    continue
                
                if case == 1: # Unconstrained GP model
                    
                    # Run active learning algorithm
                    model, x_train, max_err, rmse, r2, total_time = gp_active_learning(city, case, trial, scaler, numcores_input=numcores_input,
                                                                                            is_parallel=True, 
                                                                                            unconstrained=True, first_deriv=False, 
                                                                                            second_deriv=False, error_tol=error_tol, count_tol=count_tol)
                    
                    # Compute accuracy against grid of points
                    y_full, _ = model.predict(x_data_full)
                    y_full_actual = pd.read_csv(os.path.join('generate_results_4d','input_data_4d',f'Simulated_cases_{city}.csv')).to_numpy()[:,-1]
                    rmse_full = np.sqrt(mean_squared_error(y_full_actual,y_full))
                    r2_full = r2_score(y_full_actual,y_full) 
                    max_err_full = max_error(y_full_actual,y_full)
                    
                    # Compute accuracy against LHS selected testing set
                    y_ref, _ = model.predict(scaler.transform(x_ref))
                    rmse = np.sqrt(mean_squared_error(y_ref_actual,y_ref))
                    r2 = r2_score(y_ref_actual,y_ref) 
                    max_err = max_error(y_ref_actual,y_ref)
                    
                    # Store surrogate performance metrics
                    stats_table.loc[ii,:] = ['Unconstrained GP', x_train.shape[0], total_time, rmse, max_err, r2, rmse_full, max_err_full, r2_full]
                    
                    print('Unconstrained GP Finished')
                    
                elif case == 2: # CGP w/ boundedness and monotonicity
                    
                    # Run active learning algorithm
                    model, x_train, max_err, rmse, r2, total_time = gp_active_learning(city, case, trial, scaler, numcores_input=numcores_input,
                                                                                            is_parallel=True, 
                                                                                            unconstrained=False, first_deriv=True, 
                                                                                            second_deriv=False, error_tol=error_tol, count_tol=count_tol)
                    
                    # Compute accuracy against grid of points
                    y_full, _ = model.predict(x_data_full)
                    y_full_actual = pd.read_csv(os.path.join('generate_results_4d','input_data_4d',f'Simulated_cases_{city}.csv')).to_numpy()[:,-1]
                    rmse_full = np.sqrt(mean_squared_error(y_full_actual,y_full))
                    r2_full = r2_score(y_full_actual,y_full) 
                    max_err_full = max_error(y_full_actual,y_full)
                    
                    # Compute accuracy against LHS selected testing set
                    y_ref, _ = model.predict(scaler.transform(x_ref))
                    rmse = np.sqrt(mean_squared_error(y_ref_actual,y_ref))
                    r2 = r2_score(y_ref_actual,y_ref) 
                    max_err = max_error(y_ref_actual,y_ref)
                    
                    # Store surrogate performance metrics
                    stats_table.loc[ii,:] = ['CGP Bound + Mono', x_train.shape[0], total_time, rmse, max_err, r2, rmse_full, max_err_full, r2_full]
                    
                    print('CGP Bound + Mono Finished')
                    
                elif case == 3: # CGP w/ boundedness, monotonicity, and concavity
                
                    # Run active learning algorithm              
                    model, x_train, max_err, rmse, r2, total_time = gp_active_learning(city, case, trial, scaler, numcores_input=numcores_input,
                                                                                            is_parallel=True, 
                                                                                            unconstrained=False, first_deriv=True, 
                                                                                            second_deriv=True, error_tol=error_tol, count_tol=count_tol)
                    
                    # Compute accuracy against grid of points
                    y_full, _ = model.predict(x_data_full)
                    y_full_actual = pd.read_csv(os.path.join('generate_results_4d','input_data_4d',f'Simulated_cases_{city}.csv')).to_numpy()[:,-1]
                    rmse_full = np.sqrt(mean_squared_error(y_full_actual,y_full))
                    r2_full = r2_score(y_full_actual,y_full) 
                    max_err_full = max_error(y_full_actual,y_full)
                    
                    # Compute accuracy against LHS selected testing set
                    y_ref, _ = model.predict(scaler.transform(x_ref))
                    rmse = np.sqrt(mean_squared_error(y_ref_actual,y_ref))
                    r2 = r2_score(y_ref_actual,y_ref) 
                    max_err = max_error(y_ref_actual,y_ref)
                    
                    # Store surrogate performance metrics
                    stats_table.loc[ii,:] = ['CGP Bound + Mono + Concav', x_train.shape[0], total_time, rmse, max_err, r2, rmse_full, max_err_full, r2_full]
                    
                    print('CGP Bound + Mono + Concav Finished')                                  
                    
                elif case == 4: # Grid with RBF interpolant
                
                    # If grid-based interpolant has already been run and its results stored, continue
                    if 'Grid RBF' in stats_table.Model.unique(): 
                        grid_ok = True
                        continue
                
                    # Create interpolant object
                    interpolant = RBFInterpolator
                
                    # Calculate function at grid points and train interpoland
                    model, x_train, max_err, rmse, r2, total_time, nan_idx = interpolant_from_grid(city, interpolant, x_data_full, scaler, numcores_input=numcores_input, is_parallel=True)
                    
                    # Store surrogate performance metrics
                    stats_table.loc[ii,:] = ['Grid RBF', x_train.shape[0], total_time, rmse, max_err, r2, np.nan, np.nan, np.nan]
                    
                    print('Grid RBF Finished')
                    
                    grid_ok = True # Grid-based interpolant finished, change flag to True
                
                ii += 1 # Update counter
            
                # Save table with surrogates accuracy metrics
                stats_table.to_excel(filename)
        
    # Once all trials of all surrogates have been run, generate violin plots
    generate_violin_plots()