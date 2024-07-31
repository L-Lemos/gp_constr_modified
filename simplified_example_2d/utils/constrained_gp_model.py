# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:08:28 2023

@author: leonardo.lacerda
"""

import os, sys
os.environ['R_HOME'] = os.path.join('C:',os.sep,'R-4.3.2')

this_dir = os.getcwd()

from cgp_lib.model import GPmodel, Constraint
from cgp_lib.kern import kernel_RBF

from rpy2 import robjects
R = robjects.r
import gc

sys.path.append(os.getcwd())
os.chdir(this_dir)

import numpy as np

# In[]

def constant_function(val):
    import numpy as np
    """ Return the constant function"""
    def fun(x):
        return np.array([val]*x.shape[0])
    
    return fun

class constrained_gp_model():
    
    
    def __init__(self,input_dim,unconstrained=False):
        
        self.input_dim = input_dim  
        self.unconstrained = unconstrained
        
        # Get kernel and create model
        ker = kernel_RBF(variance = 1, lengthscale = [1]*self.input_dim)        
        model = GPmodel(kernel = ker, likelihood = 1E-6, mean = 0)

        if not(self.unconstrained):
            # Add constraints
            # constr_bounded MUST BE A SINGLE CONSTRAINT AND CANNOT BE None. 
            # THERE MUST BE A BOUNDEDNESS CONSTRAINT
            constr_bounded = Constraint(LB = constant_function(0), UB = constant_function(1))
            model.constr_bounded = constr_bounded  
            
            constr_deriv = [
                Constraint(LB = constant_function(0), UB = constant_function(float('Inf'))),
                Constraint(LB = constant_function(0), UB = constant_function(float('Inf')))
            ]
            model.constr_deriv = constr_deriv
      
            constr_deriv_2 = [
                Constraint(LB = constant_function(-float('Inf')), UB = constant_function(0)),
                Constraint(LB = constant_function(-float('Inf')), UB = constant_function(0))
            ]
            model.constr_deriv_2 = constr_deriv_2       
           
            model.constr_likelihood = 1E-6
        
        self.model = model
        
        
    
    def train_model(self, x_train, y_train, virtual_points, sigma_bounds=[(1E-6,1E-4)]):
        
        model = self.model
        
        # Fit the regressor on initial training dataset
        model.X_training = x_train
        model.Y_training = y_train        
        
        # Add virtual points in a "domain-filling" strategy       
        
        for xv in virtual_points:      
            
            if model.constr_bounded is not None:
                model.constr_bounded.add_XV(xv)
            
            if model.constr_deriv is not None:
                for cc, constr in enumerate(model.constr_deriv):
                    if constr is not None:
                        model.constr_deriv[cc].add_XV(xv)
            
            if model.constr_deriv_2 is not None:
                for cc, constr in enumerate(model.constr_deriv_2):
                    if constr is not None:
                        model.constr_deriv_2[cc].add_XV(xv)
            
        # Reset computations depending on XV
        model.reset_XV()        
        
        # Optimize model
        # DECIDED FOR UNCONSTRAINED OPTIMIZATION, LITTLE GAIN FROM CONSTRAINED OPT, PLUS TOO MUCH COMPUTATION TIME
        model.optimize(include_constraint = False, fix_likelihood = False, bound_min = 0.5, sigma_bounds=sigma_bounds)
        
        print(f'Model Hyperparameters: {[model.likelihood] + list(model.kernel.get_params())}')
        
        gc.collect() # Garbage collection
        R('rm(list = ls(all.names=TRUE))') # Refresh R to prevent memory leaks
        R('gc()')# R garbage collection 
        
        self.model = model  
        
        

    def predict(self,x_test):                  
        
        model = self.model
        
        if self.unconstrained:
            mean, var = model.calc_posterior_unconstrained(x_test, full_cov=False) 
        else:
            mean, var, perc, mode, samples, sampling_time = model.calc_posterior_constrained(x_test, num_samples=1000)  
            
        mean = np.array(mean).flatten()
        var = np.array(var).flatten()
        
        R('warnings()')# Show warnings from R during sampling
        
        return mean, var