# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 05:38:56 2023

@author: leonardo.lacerda
"""

# Import libraries

import sys
import os
os.environ['R_HOME'] = os.path.join('C:',os.sep,'Users','leonardo.lacerda','AppData','Local','Programs','R','R-4.3.1')

this_dir = os.getcwd()

from GPConstr_EXPERIMENTO_CONVEX.model import GPmodel, Constraint
from GPConstr_EXPERIMENTO_CONVEX.kern import kernel_RBF

sys.path.append(os.getcwd())
os.chdir(this_dir)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# In[Constant function for constraint specification]

def constant_function(val):
    """ Return the constant function"""
    def fun(x):
        return np.array([val]*x.shape[0])
    
    return fun

# In[Create model]

class constrained_gp_model():
    
    def __init__(self,input_dim):
        
        self.input_dim = input_dim
        
        # Get kernel and create model
        ker = kernel_RBF(variance = 1, lengthscale = [1]*self.input_dim)

        
        model = GPmodel(kernel = ker, likelihood = 1, mean = 0) # With noise
        
        # Add constraints
    
        constr_bounded = Constraint(LB = constant_function(0), UB = constant_function(1))
    
        constr_deriv = [
            Constraint(LB = constant_function(0), UB = constant_function(float('Inf'))),
            Constraint(LB = constant_function(0), UB = constant_function(float('Inf')))
        ]
    
        constr_deriv_2 = [
            Constraint(LB = constant_function(-float('Inf')), UB = constant_function(0))      
        ]
    
        # Add constraints to model
        model.constr_deriv = constr_deriv
        model.constr_deriv_2 = constr_deriv_2
        model.constr_bounded = constr_bounded
        model.constr_likelihood = 1E-6
        
        self.model = model
    
    def train_model(self, x_train, y_train, opt_method):
        
        model = self.model
        
        # Fit the regressor on initial training dataset
        model.X_training = x_train
        model.Y_training = y_train
        
        # Optimize model considering no constraints
        model.optimize(include_constraint = False, fix_likelihood = False, opt_method = opt_method)
        
        # Search for a suitable set of virtual observation locations where the constraint is imposed -- finite search
    
        # First with a high p_target to ensure that we get at least one point
        Omega = np.random.uniform(size = (1000, self.input_dim)) # Candidate set
        df, num_pts, pc_min = model.find_XV_subop(p_target = 0.99, Omega = Omega, sampling_alg = 'minimax_tilting', num_samples = 1000,
                                 max_iterations = 1, print_intermediate = False)
    
        # Then we run multiple iterations with p_target = 0.7 
        Omega = np.random.uniform(size = (1000, self.input_dim)) # Candidate set
        df, num_pts, pc_min = model.find_XV_subop(p_target = 0.7, Omega = Omega, sampling_alg = 'minimax_tilting', num_samples = 1000,
                                 max_iterations = 200, print_intermediate = False)
        
        # Optimize with constraints    
        # model.optimize(include_constraint = True, fix_likelihood = False, opt_method = 'L-BFGS-B', conditional = True, bound_min = 0.8)
        
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot(df.loc[:,'Xv[1]'], df.loc[:,'Xv[2]'], np.zeros((df.shape[0],1)).flatten(), 'dr')
        # ax.set_xlim((0, 1))
        # ax.set_ylim((0, 1))  
        # plt.title(f'Virtual Points Sampled')
        # plt.show()
        
        self.model = model
    
    def predict(self,x_test):        
        
        model = self.model
        
        mean, var, perc, mode, samples, sampling_time = model.calc_posterior_constrained(x_test) #, compute_mode = False, num_samples = 10000, save_samples = 30, algorithm = 'minimax_tilting', resample = False)
        mean = np.array(mean).flatten()
        lower = perc[0]
        upper = perc[2]
        var = np.array(var).flatten()
        
        return mean, var, lower, upper

# In[Load CSV data 1-D]

# df = pd.read_csv('1_d_data.csv', index_col=0)

# In[Load CSV data 2-D]

# Import data
df = pd.read_csv('Simulated_cases_Iguatu.csv', index_col=0)
idx = df.col_orient == df.col_orient.unique()[0]
idx = idx & (df.col_beta_sumfac == df.col_beta_sumfac.unique()[0])
idx = idx & (df.Q_year_MWh == df.Q_year_MWh.unique()[1])
idx = idx & (df.T_demand == df.T_demand.unique()[4])
idx = idx & (df.T_return == df.T_return.unique()[1])
df_full = df.copy()
df = df.loc[idx, ['A_sf_m2', 'r_st_m3_per_m2', 'SOLFRAC']]
df = df.reset_index(drop=True)

# Get two random points to start training
first_points = np.random.randint(df.shape[0],size=(2)).tolist()
x_train = df.iloc[first_points,0:2].values
y_train = df.iloc[first_points,2].values.flatten()
a_domain = np.linspace(0, 1000, 51)
r_domain = np.linspace(0, 0.08, 51)
x_domain_grid = np.meshgrid(a_domain, r_domain)
x_domain = np.concatenate((x_domain_grid[0].flatten().reshape(-1, 1), x_domain_grid[1].flatten().reshape(-1, 1)),axis=1) 

# Transform variables to normalized values
scaler = MinMaxScaler().fit([[0.,0.],[1000,0.08]])    
x_train = scaler.transform(x_train)
x_domain = scaler.transform(x_domain)
df.iloc[:,0:2] = scaler.transform(df.iloc[:,0:2].values)

# In[Create and train model]

# Copy of df indexes
df_idx = df.index.tolist()

# Remove already selected points fron list of indexex
df_idx.remove(first_points[0])
df_idx.remove(first_points[1])

# Iterative loop to create models with increasing number of training points

while x_train.shape[0] <= 10:
    
    # Create model
    model = constrained_gp_model(x_train.shape[1])
    
    # Train
    model.train_model(x_train,y_train,opt_method='Nelder-Mead')
    
    # Get mean and variance along domain after fitting model
    y_mean, y_var, y_lower, y_upper = model.predict(x_domain) 
    
    # Add new point
    new_point = np.random.choice(df_idx,size=1).tolist()
    x_train = np.concatenate([x_train,df.iloc[new_point,0:2].values])
    y_train = np.concatenate([y_train,df.iloc[new_point,2].values])
    
    # Update list of points that can be sampled
    df_idx.remove(new_point[0])