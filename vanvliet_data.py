import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from netCDF4 import Dataset
import statsmodels.api as sm


datadir = '../isimip-ps/data/processed/globalmean_annual'

scenarios = ['rcp26', 'rcp85']
decades = ['2020s', '2050s']


def fit(x, beta_t, beta_t2):
    yhat = beta_t*x + beta_t2*x**2
    return yhat

#%%

esms = ['MIROC5', 'IPSL-CM5A-LR', 'HadGEM2-ES', 'GFDL-ESM2M']

esms_ft = ['MIROC-ESM-CHEM', 'IPSL-CM5A-LR', 'HadGEM2-ES', 
        'GFDL-ESM2M', 'NorESM1-M']

datadir_ft = 'vanVliet2016/FastTrack_tas_globalmean'

temp_data_no_pi = {}

for scen in scenarios:
    temp_data_no_pi[scen] = {}
    for decade in decades:
        temp_data_no_pi[scen][decade] = {}
        
        temp_data_no_pi[scen][decade]['ISIMIP2b'] = {}
        temp_data_no_pi[scen][decade]['FastTrack'] = {}


        
        for esm in esms_ft:
            
            esm_file = glob.glob(f'{datadir_ft}/{esm}_{scen}_r1i1p1_tas_*.globmean.nc')[0]
            
            data_in = Dataset(f'{esm_file}')
            data_tas = data_in.variables['tas'][:]
            
            y1_file = int(esm_file.split("-20")[0].split("_")[-1])
            
            offset = 2006-y1_file
            
            temp_data_no_pi[scen][decade]['FastTrack'][esm] = data_tas[offset:,0,0]

        for esm in esms:
            
            esm_file = glob.glob(f'{datadir}/tas_annual_{esm}'
                        f'_{scen}_r1i1p1_EWEMBI_2006-*.csv')[0]
            
            df_in = pd.read_csv(esm_file)
            
            df_tas = df_in[(2006 <= df_in['YEARS']) &
                           (df_in['YEARS'] < 2100)]['WORLD'].values
            
            temp_data_no_pi[scen][decade]['ISIMIP2b'][esm] = df_tas
            
esms_colors = {
    'MIROC-ESM-CHEM':'red', 
    'IPSL-CM5A-LR':'blue', 
    'HadGEM2-ES':'orange',
    'GFDL-ESM2M':'purple', 
    'NorESM1-M':'green',
    'MIROC5':'red', 
    }           

linestyles = {
    'ISIMIP2b':['solid', 1],
    'FastTrack':['dashed', 1],    
    }

time = 2006 + np.arange(94)

for scen in scenarios:
    for decade in decades:
        for gen in ['ISIMIP2b','FastTrack']:
            for esm in temp_data_no_pi[scen][decade][gen].keys():
                
                plt.plot(time, temp_data_no_pi[scen][decade][gen][esm],
                     color=esms_colors[esm], linestyle=linestyles[gen][0],
                     linewidth=linestyles[gen][1])
                

handles = []

for esm in esms_ft:
    handles.append(
        Line2D([0], [0], color=esms_colors[esm], label=esm),
        ) 

for gen in ['ISIMIP2b','FastTrack']:                   
    handles.append(
        Line2D([0], [0], color='grey', linestyle = linestyles[gen][0], 
                   linewidth=linestyles[gen][1], label=gen),
        )     

plt.legend(handles=handles)
plt.ylabel('GMST')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig('plots/gmst_no_pi.png', dpi=100)
# plt.clf()

# CONCLUSION: FastTrack and ISIMIP2b are not the same data, even for the 
# same ESMs. But they are all close ish in present-day. So we can try and 
# take the pi to pd dif in the ISIMIP2b data, and apply it to the FastTrack
# runs. 

#%%


temp_data = {}

for scen in scenarios:
    temp_data[scen] = {}
    for decade in decades:
        temp_data[scen][decade] = np.zeros(len(esms_ft))

        y1 = int(decade[:-1])
        y2 = y1 + 9
        
        isimip_difs = np.full((len(esms_ft)-1), np.nan)
        for e_i, esm in enumerate(esms_ft):
            
            if esm == 'MIROC-ESM-CHEM':
                esm_isimip = 'MIROC5'
            
            else:               
                esm_isimip = esm


            # Get ISIMIP2b 1961-2000 - 1861-1900 offset
            if esm == 'NorESM1-M': # NEEDS TO BE LAST IN LIST
                isimip_dif = np.mean(isimip_difs)
            else:
    
                isimip2b_hist_file = glob.glob(f'{datadir}/tas_annual_{esm_isimip}'
                            f'_historical_r1i1p1_EWEMBI_1861-*.csv')[0]
                
                df_hist = pd.read_csv(isimip2b_hist_file)
                
                df_pi = np.mean(df_hist[(1861 <= df_hist['YEARS']) &
                               (df_hist['YEARS'] <= 1900)]['WORLD'].values)
                
                df_late20th = np.mean(df_hist[(1961 <= df_hist['YEARS']) &
                               (df_hist['YEARS'] <= 2000)]['WORLD'].values)
                
                isimip_dif = df_late20th - df_pi
                isimip_difs[e_i] = isimip_dif
            
                        
            
            # Get FastTrack hist data
            esm_hist_file = glob.glob(f'{datadir_ft}/{esm}'
                        f'_historical_r1i1p1_tas_1960*.nc')[0]
            
            data_in = Dataset(f'{esm_hist_file}')
            data_tas = data_in.variables['tas'][:]
            
            fasttrack_raw_hist = data_tas[:,0,0]
            
            
            # Get FastTrack future data
            esm_file = glob.glob(f'{datadir_ft}/{esm}_{scen}_r1i1p1_tas_*.globmean.nc')[0]
            
            data_in = Dataset(f'{esm_file}')
            data_tas = data_in.variables['tas'][:]
            
            fasttrack_raw_fut = data_tas[:,0,0]
            

            
            
            # Concat FastTrack data, offset
            fasttrack_raw = np.concatenate((fasttrack_raw_hist, fasttrack_raw_fut))
                
            fasttrack_corr = fasttrack_raw - np.mean(fasttrack_raw[:40]) + isimip_dif
            
            fasttrack_corr_dec = np.mean(fasttrack_corr[y1-1960:y2-1960])
            
            
            temp_data[scen][decade][e_i] = fasttrack_corr_dec
            
#%%

energy_change = {

    'Thermal':{
    'Baseline':{
        'rcp26':{
            '2020s':[-7.4, -5.8, -4.5],
            '2050s':[-9.9, -7, -4.3],
            },
        'rcp85':{
            '2020s':[-6.6, -5.3, -3.3],
            '2050s':[-14.5, -12.1, -8.6],
            
            }
        },
    # 'Adaptation':{
    #     'rcp26':{
    #         '2020s':3.5,
    #         '2050s':1.52,
    #         },
    #     'rcp85':{
    #         '2020s':3,
    #         '2050s':-3.26,
            
    #         }
    #     }
        
    },
   
    'Hydro':{
    'Baseline':{
        'rcp26':{
            '2020s':[-2.7, -1.7, -0.6],
            '2050s':[-2.5, -1.2, 1.0],
            },
        'rcp85':{
            '2020s':[-2.6, -1.9, -0.8],
            '2050s':[-5.2, -3.6, -2.4],
            
            }
        },
    # 'Adaptation':{
    #     'rcp26':{
    #         '2020s':1.6,
    #         '2050s':2.2,
    #         },
    #     'rcp85':{
    #         '2020s':1.5,
    #         '2050s':-0.3,
            
    #         }
    #     }
        
    }
}

#%%

levels = ['low', 'med', 'high']

params = {}
response = 'Baseline'
params[response] = {}



for l_i, level in enumerate(levels):
    params[response][level] = {}
    for source in energy_change.keys():
    
        temps = []
        impacts = []
        for scen in scenarios:
            for decade in decades:
                temps.append(np.mean(temp_data[scen][decade]))
                impacts.append(energy_change[source][response][scen][decade][l_i])
        
        params_in, _ = curve_fit(
            fit, temps, impacts)
        
        params[response][level][source] = params_in
        
        

#%%

temps_plot = np.linspace(0, 4.5, 100)

colors = {
    'Adaptation':'blue', 
    'Baseline':'red',
    }

markers = {
    '2020s':'x', 
    '2050s':'o',
    }

for source in energy_change.keys():
    for scen in scenarios:
        for decade in decades:
            
            # plt.scatter(np.mean(temp_data[scen][decade]), 
            #     energy_change[source]['Adaptation'][scen][decade],
            #     color=colors['Adaptation'], marker=markers[decade])
            
            for l_i, level in enumerate(levels):
                    
                plt.scatter(np.mean(temp_data[scen][decade]), 
                    energy_change[source]['Baseline'][scen][decade][l_i],
                    color=colors['Baseline'], marker=markers[decade])
                

            
            
                 # for e_i, esm in enumerate(esms):
       
                 #    plt.scatter(temp_data[scen][decade][e_i], 
                 #        energy_change[source][response][scen][decade],
                 #        color=colors[response], marker=markers[decade])
    
    handles = []
    
    for decade in decades:
        handles.append(
            Line2D([0], [0], label=f'{decade}', marker=markers[decade], 
                   color='black', linestyle=''),
            )    
    
    for response in ['Baseline', 'Adaptation']:
        handles.append(
            Line2D([0], [0], label=f'{response}', marker='s', 
                   color=colors[response], linestyle=''),
            ) 
        
        
    for level in levels:
    
        plt.plot(temps_plot, fit(temps_plot, *params['Baseline'][level][source]), 
                 color=colors['Baseline'])
            
        
    
    plt.legend(handles=handles)
    plt.xlabel('GMST cf 1861-1900')
    plt.ylabel(f'% damages on {source} power plants')
    plt.title(f'{source}')
    plt.tight_layout()
    plt.savefig(f'plots/{source}_gmst_mmm.png', dpi=100)
    plt.clf()

#%%

for source in energy_change.keys():
    
    for scen in scenarios:
        for decade in decades:
            
            plt.scatter(np.mean(temp_data[scen][decade]), 
                energy_change[source]['Adaptation'][scen][decade
               ] - energy_change[source]['Baseline'][scen][decade],
                color='black', marker=markers[decade])
    
    plt.xlabel('GMST cf 1861-1900')
    plt.ylabel('Difference in % damages \n Adaptation - Baseline')
    plt.tight_layout()
    plt.savefig(f'plots/{source}_adaptation_dif.png', dpi=100)
    plt.clf()
#%%


levels = ['low', 'med', 'high']
response='Baseline'
params_all_levels = {}

for source in energy_change.keys():
    print(source)
    params_all_levels[source] = {}

    temps = []
    impacts = []
    for l_i, level in enumerate(levels):

        for scen in scenarios:
            for decade in decades:
                temps.append(np.mean(temp_data[scen][decade]))
                impacts.append(energy_change[source][response][scen][decade][l_i])
        
    params_in, params_cov = curve_fit(
        fit, temps, impacts)
    
    print(f'curve_fit params {params_in}')
    print(f'curve_fit errors {np.sqrt(np.diag(params_cov))} \n')

    params_all_levels[source]['Parameters'] = params_in 
    params_all_levels[source]['Errors'] = np.sqrt(np.diag(params_cov))
          
    
    data = {"impacts":impacts, "temps":temps}
    model = sm.ols(formula = 'impacts ~ temps + np.power(temps, 2) -1', data = data)
    results = model.fit()
    results.params
    
    
    print(f'sm.ols params {results.params}')
    print(f'sm.ols errors {results.bse} \n \n')

         
