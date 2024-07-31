# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:21:56 2023

@author: leonardo.lacerda
"""

import os
os.environ['R_HOME'] = os.path.join('C:',os.sep,'R-4.3.2')

import numpy as np
import pandas as pd
from sklearn.metrics import max_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time
from scipy.stats import qmc

import matplotlib.pyplot as plt

from simplified_example_2d import constrained_gp_model, run_trnsys_sim_2d, maximin_dist

from rpy2 import robjects
R = robjects.r

plt.close('all')

# In[User inputs for this example]

unconstrained = False # Flag indicating whether to train an unconstrained CGP or a fully constrained CGP with boundedness, monotonicity and concavity constraints

use_lhs_for_virtual = True # Whether to use LHS to select virtual points. If False, maximin will be used
n_virtual = 30 # Number of virtual points to be used

use_lhs_for_train = True # Whether to use LHS to select training points. If False, maximin will be used
n_train = 10 # Number of training points to be used

# In[Train model]

# Initialize LHS sampler
sampler = qmc.LatinHypercube(d=2)

# Set of virtual points --> Can be chosen via LHS or maximin
if use_lhs_for_virtual:  
    virtual_points = sampler.random(n=n_virtual)
else:
    virtual_points = maximin_dist(None, n_virtual, 2)  

# Training points --> Can be chosen via LHS or maximin   
if use_lhs_for_train:                                          
    x_train = sampler.random(n=n_train)
else:
    x_train = maximin_dist(None, n_train, 2) 

# Transform variables to normalized values
domain_limits = [[0.,0.],[950,0.08]]
scaler = MinMaxScaler().fit(domain_limits)    

# Generate inputs for TRNSYS
x_sample_trnsys = scaler.inverse_transform(x_train)    
col_area = x_sample_trnsys[:,0].tolist()
storage = x_sample_trnsys[:,1].tolist()

print('Running interpolant with TRNSYS emulation')
start_time = time.time()
solfrac = run_trnsys_sim_2d(col_area,storage)
solfrac = np.array(solfrac).reshape(x_train.shape[0],1).flatten()

x_train = x_train[~np.isnan(solfrac),:]
solfrac = solfrac[~np.isnan(solfrac)]

y_train = solfrac
print(f'{time.time() - start_time:.2f} for TRNSYS simulation')

# Create model
model = constrained_gp_model(x_train.shape[1], unconstrained=unconstrained)

# Train
model.train_model(x_train,y_train,virtual_points)

# In[Get model estimates before plotting] 

# Array with fine grid on X-domain, used for plotting of surface
a_domain = np.linspace(0, domain_limits[1][0], 11) #np.linspace(0, 1000, 51)
r_domain = np.linspace(0, domain_limits[1][1], 11) #np.linspace(0, 0.08, 51)
x_domain_grid = np.meshgrid(a_domain, r_domain)
x_domain = np.concatenate((x_domain_grid[0].flatten().reshape(-1, 1), x_domain_grid[1].flatten().reshape(-1, 1)),axis=1) 
x_domain = scaler.transform(x_domain)

# Get estimates from CGP surrogate across domain                                   
y_mean, y_var = model.predict(x_domain)

# Generate surfaces of CGP mean and variance
a_axis = np.unique(x_domain[:, 0])
r_axis = np.unique(x_domain[:, 1])

a_grid = x_domain[:, 0].reshape(a_axis.shape[0], r_axis.shape[0])
r_grid = x_domain[:, 1].reshape(a_axis.shape[0], r_axis.shape[0])
y_mean_grid = y_mean.reshape(a_axis.shape[0], r_axis.shape[0])
y_var_grid = y_var.reshape(a_axis.shape[0], r_axis.shape[0])

# Get reference grid of data points
df = pd.read_csv(os.path.join('simplified_example_2d','input_data','reference_grid.csv')).to_numpy()
x_data_full = df[:, 0:2].reshape((-1, 2))
y_data_full = df[:, -1] 
# Add line with zeros at zero solar collector area
temp = np.linspace(0,x_data_full[:, 1].max(),5).reshape(-1, 1).repeat(2, axis=1)
temp[:, 0] = 0
x_data_full = np.concatenate([temp, x_data_full])
y_data_full = np.concatenate([temp[:, 0], y_data_full]) 
x_data_full = scaler.transform(x_data_full)

# Also get estimates of CGP at training points and at reference grid
y_pred, _ = model.predict(x_train)
y_pred_full, _ = model.predict(x_data_full)

# In[Start plotting]

# Azimuth and elevation of plot view
azim = -127.69
elev = 14.90

# Plot of CGP surface, overlaid with reference grid and training points

fig = plt.figure()
ax = plt.axes(projection='3d', computed_zorder=True) # Manually set zorder?
ax.view_init(azim=azim,elev=elev)

# Check if surface of CGP mean must be plotted above or below training points
idx_pred = y_train > y_pred
if sum(idx_pred) > 0:
    ax.plot(x_train[idx_pred,0], x_train[idx_pred,1], y_train[idx_pred], 'dr',zorder=30)
if sum(~idx_pred):
    ax.plot(x_train[~idx_pred,0], x_train[~idx_pred,1], y_train[~idx_pred], 'dr',zorder=10)

    # Check if surface of CGP mean must be plotted above or below reference grid
idx_full = y_data_full > y_pred_full
if sum(idx_full) > 0:
    ax.plot(x_data_full[idx_full,0], x_data_full[idx_full,1], y_data_full[idx_full], 'xk', zorder=40)
if sum(~idx_full) > 0:
    ax.plot(x_data_full[~idx_full,0], x_data_full[~idx_full,1], y_data_full[~idx_full], 'xk', zorder=0)

# Plot surface of CGP mean
gp = ax.plot_surface(a_grid, r_grid, y_mean_grid, alpha=0.9, zorder=20)

# Plot training points
tp = ax.scatter(x_train[0,0],x_train[0,1],y_train[0],marker='d',color='red')

# Plot points of reference grid
fp = ax.scatter(x_data_full[0,0],x_data_full[0,1],y_data_full[0],marker='x',color='black')

# Final details of plot, legend, axis, and labels
gp._edgecolors2d = gp._edgecolor3d
gp._facecolors2d = gp._facecolor3d

plt.legend([tp, fp, gp], ['Training Points','Reference Data','GP Regression'],loc='upper center',ncol=3)

ax.set_xlim((-0.01, max(a_axis)))
ax.set_ylim((-0.01, max(r_axis)))
ax.set_zlim((-0.01, 1.2))
ax.set_xlabel('Area')
ax.set_ylabel('Storage')
ax.set_zlabel('Sol. Frac.')

# Get error metrics
y_grid_pred, _ = model.predict(x_data_full)          
max_err = max_error(y_data_full, y_grid_pred)
r2 = r2_score(y_data_full, y_grid_pred)
rmse = np.sqrt(mean_squared_error(y_data_full, y_grid_pred))

fig.suptitle('Error metrics of GP model against reference grid (black markers)')
plt.title(f'RMSE = {rmse:.2e}, R2 = {r2:.3f}, Max. Err. = {max_err:.3f}') 

plt.savefig(os.path.join('simplified_example_2d','outputs','Trained CGP.png'),bbox_inches='tight',dpi=300)

# Plot Virtual Points
fig = plt.figure()
ax = plt.axes(projection='3d', computed_zorder=False)  
ax.view_init(azim=azim,elev=elev)       
ax.plot(virtual_points[:,0], virtual_points[:,1], np.zeros((virtual_points.shape[0])), 'dg')      
ax.set_xlim((-0.01, max(a_axis)))
ax.set_ylim((-0.01, max(r_axis)))
ax.set_zlim((-0.01, 1.2))
ax.set_xlabel('Area')
ax.set_ylabel('Storage')
ax.set_zlabel('Sol. Frac.')
plt.title('Selected virtual points')
plt.savefig(os.path.join('simplified_example_2d','outputs','Selected virtual points.png'),bbox_inches='tight',dpi=300)   