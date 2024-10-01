import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from netCDF4 import Dataset

datadir = 'Yalew2020/FastTrack_tas_globalmean'

scenarios = ['rcp26', 'rcp85']
decades = ['2020s', '2050s']
esms = ['MIROC-ESM-CHEM', 'IPSL-CM5A-LR', 'HadGEM2-ES', 
        'GFDL-ESM2M', 'NorESM1-M']

#%%
temp_data = {}

for scen in scenarios:
    temp_data[scen] = {}
    for decade in decades:
        temp_data[scen][decade] = np.zeros(len(esms))

        y1 = int(decade[:-1])
        y2 = y1 + 9
        
        idx1 = y1 - 2006
        idx2 = y2 - 2006

        
        for e_i, esm in enumerate(esms):
            
            esm_file = f'{esm}_{scen}_r1i1p1_tas_2006-2099.globmean.nc'
            
            data_in = Dataset(f'{datadir}/{esm_file}')
            
            data_tas = data_in.variables['tas'][:]
            
            
            tas_dec = np.mean(data_tas[idx1:idx2])
            
            
            # need from 1861...
            esm_hist_file = f'{esm}_historical_r1i1p1_tas_2006-2099.globmean.nc'

            
            temp_data[scen][decade][e_i] = tas_dec - []
            
#%%

thermal_change = {
    'Baseline':{
        'rcp26':{
            '2020s':-5.8,
            '2050s':-7,
            },
        'rcp85':{
            '2020s':-5.3,
            '2050s':-12.1,
            
            }
        },
    'Adaptation':{
        'rcp26':{
            '2020s':3.5,
            '2050s':1.52,
            },
        'rcp85':{
            '2020s':3,
            '2050s':-3.26,
            
            }
        }
        
    }

#%%

colors = {
    'Adaptation':'blue', 
    'Baseline':'red',
    }

markers = {
    '2020s':'x', 
    '2050s':'o',
    }

for scen in scenarios:
    for decade in decades:
        for response in thermal_change.keys():
            
            plt.scatter(np.mean(temp_data[scen][decade][e_i]), 
                thermal_change[response][scen][decade],
                color=colors[response], marker=markers[decade])


