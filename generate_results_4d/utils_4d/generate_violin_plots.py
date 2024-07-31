# -*- coding: utf-8 -*-
# Import necessary libraries, close any existing figure
import os
import pandas as pd
import matplotlib.pyplot as plt

# In[]

def generate_violin_plots():

    plt.close('all')

    # Metrics to be included into violin plot
    metrics = ['Num Points', 'Total Time', 'RMSE', 'Max Error']
    
    # In this example, only Iguatu data is available
    for city in ['Iguatu']:
    
        # Reference data for comparing generated models
        filename = os.path.join('generate_results_4d','output_data_4d',f"stats_table_{city.lower().replace(' ','_')}.xlsx")
        
        try:
            df = pd.read_excel(filename, index_col=0)
        except:
            continue
        
        models = df.Model.unique()
        
        model_labels = list()
        for model in models:
            if model == 'Unconstrained GP':
                model_labels.append('Unconstr. GP')
            elif model == 'CGP Bound + Mono + Concav':
                model_labels.append('CGP Bound + \n Mono + Concav')
            elif model == 'CGP Bound + Mono':
                model_labels.append('CGP Bound + \n Mono')
        
        plt.close('all')
        
        fig, axes = plt.subplots(2,2, ) # create figure and axes
        fig.set_figheight(10)
        fig.set_figwidth(10)        
        
        for ii, metric in enumerate(metrics):
            temp = pd.DataFrame()
            for model in models:
                if 'grid' in model.lower():
                    grid_val = df.loc[df.Model == model, metric].tolist()[0]
                    continue
                try:
                    temp.loc[:,model] = df.loc[df.Model == model, metric].tolist()
                except:
                    xdf = df.loc[df.Model == model, metric].tolist()
                    if len(xdf) > temp.shape[0]:
                        xdf = xdf[0:temp.shape[0]]
                    else:
                        temp = temp.iloc[0:len(xdf),:]
                    temp.loc[:,model] = xdf
                
            
            temp.columns = model_labels        
            
            temp = temp.loc[:,['Unconstr. GP', 'CGP Bound + \n Mono', 'CGP Bound + \n Mono + Concav']] # Make sure labels are in uGP, CGP Mono, CGP Mono+Concav order
    
            ax = axes.flatten()[ii]
            max_val = max(grid_val, temp.max().max())
            ax.set_ylim([0, 1.1*max_val])
            x_entries = list(range(1,len(models)))
            ax.plot(x_entries, [grid_val for x in x_entries], 'r', linewidth=2)
            ax.violinplot(temp, showmeans=True)
            ax.yaxis.grid(True)
            ax.set_xticks([y+1 for y in range(len(temp.columns))], labels=temp.columns.tolist())
            ax.tick_params(axis='x',rotation=15)
    
            if metric == 'Num Points':
                ax.set_title('N° of Training Points (-)', fontsize=12, fontweight='bold')
                ax.set_ylim([0, 700])
            elif metric == 'Total Time':
                ax.set_title('Total time (s)', fontsize=12, fontweight='bold')
                ax.set_ylim([0, 15000])
            elif metric == 'RMSE':
                ax.set_title('RMSE (-)', fontsize=12, fontweight='bold')
                ax.set_ylim([0, 0.040])
            elif metric == 'Max Error':
                ax.set_title('Max. Error (-)', fontsize=12, fontweight='bold')
                ax.set_ylim([0, 0.225])
            
        fig.suptitle(city, fontsize=16, fontweight='bold')
        fig.tight_layout(pad=3.0)
        
        plt.savefig(os.path.join('generate_results_4d','output_data_4d',f'violin_plots_{city}.png'),bbox_inches='tight',dpi=300)
            
    # In[]
    
    plt.close('all')
    
    return