import os
import glob
import sys
import numpy as np
import iris
import iris.coord_categorisation
from tqdm.auto import tqdm
import pandas as pd
import requests

# this script gets the ISIMIP2b climate data we need to generate the GMST
# values for use in the regression

datadir = '../data'

esms = ['GFDL-ESM2M', 'MIROC5', 'HadGEM2-ES', 'IPSL-CM5A-LR']
scenarios = ['historical', 'rcp26', 'rcp60', 'rcp85']

#%%

# generate list of ISIMIP climate files to be downloaded

client = requests.session()

file_list = []
for esm in esms:
    for scen in scenarios:
        search = f"https://data.isimip.org/api/v1/datasets/?tree=ISIMIP2b/InputData/climate/atmosphere/{esm.lower()}/{scen}/tas"
        r = client.get(search)

        if len(r.json()['results']) > 0:
            for fi1 in np.arange(len(r.json()['results'])):
                for fi2 in np.arange(len(r.json()['results'][fi1]['files'])):
                    file_list.append(r.json()['results'][fi1]['files'][fi2]['file_url'])
                
with open('../filelists/filelist_climate.txt', 'w+') as f_out:
    for f in file_list:
        if 'EWEMBI' not in f:
            raise Exception(f"File is not ewembi bias-corrected {f}")
        if 'tasmin' not in f:
            if 'tasmax' not in f:
                f_out.write(f'{f}\n')
f_out.close()


#%%

# download ISIMIP climate files


with open('../filelists/filelist_climate.txt', 'r') as f:
    url_list = [line.rstrip() for line in f]
    
for url in tqdm(url_list, desc="Downloading files"):
    filename = url.split('/')[-1]

    # for any file: if already downloaded, skip
    # I think this is sort of like wget -c, but without any fancy resuming capability
    # or check for file size
    if os.path.exists(os.path.join(datadir, 'from_archive', filename)):
        continue

    #check if processed annual file exists
    (
        var,
        freq,
        model,
        expt,
        run,
        bias_corr,
        period_dot_nc4
    ) = filename.split('_')
    year_start, year_end = (period_dot_nc4[:4], period_dot_nc4[9:13])
    annual_filename = f'{var}_annual_{model}_{expt}_{run}_{bias_corr}_{year_start}-{year_end}.csv'
    if os.path.exists(os.path.join(datadir, 'processed', 'globalmean_annual', annual_filename)):
        # it exists, so skip rest of loop: don't re-download it
        continue

    # if none of above, download the file
    response = requests.get(url, stream=True)
    with open(os.path.join(datadir, 'from_archive',filename), mode="wb") as fileout:
        for chunk in response.iter_content(chunk_size = 500 * 1024):
            fileout.write(chunk)
            


#%%

# global annual mean ISIMIP climate data

daily_climate_filepaths = glob.glob(os.path.join(datadir, 'from_archive', '*_day_*'))

for filepath in tqdm(daily_climate_filepaths):
    filename = os.path.split(filepath)[-1]
    var, freq, model, expt, run, biascorr, timespan = filename.split('_')
    cube = iris.load_cube(filepath)
    iris.coord_categorisation.add_year(cube, 'time')
    cube_annual = cube.aggregated_by('year', iris.analysis.MEAN)
    years = np.unique(cube_annual.coord("year").points)
    yearstart = min(years)
    yearend = max(years)
    
    
    
    # Attempt to annual-mean the cube. If successful, delete the daily data
    try:
        iris.save(cube_annual, os.path.join(datadir, 'processed', 
                        'gridded_annual', f"{var}_annual_{model}_{expt}_{run}_{biascorr}_{yearstart}-{yearend}.nc"))
    except Exception:
        sys.exit(1)
    else:
        os.remove(filepath)
        

for esm in tqdm(esms, desc='ESM', position=0):
    for scenario in tqdm(scenarios, desc='Scenario', position=1, leave=False):
        annual_climate_filepaths = glob.glob(os.path.join(datadir, 'processed', 
                          'gridded_annual', f'tas_annual_{esm}_{scenario}_*.nc'))
        
        if len(annual_climate_filepaths) > 0:
            run, biascorr, timespan = annual_climate_filepaths[0].split('.')[-2].split('_')[-3:]
            cube = iris.load(annual_climate_filepaths)
            cube = cube.concatenate()[0]
            if not cube.coord('latitude').has_bounds():
                cube.coord('latitude').guess_bounds()
            if not cube.coord('longitude').has_bounds():
                cube.coord('longitude').guess_bounds()
            grid_areas = iris.analysis.cartography.area_weights(cube)

            # Global mean temperature (land + ocean 2m air SAT) 
            cube_gm = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
            yearstart = min(cube_gm.coord('year').points)
            yearend = max(cube_gm.coord('year').points)
            
            df_out = pd.DataFrame(
                columns=[],
            )
            df_out['TAS'] = cube_gm.data
            df_out.index=cube_gm.coord('year').points
            df_out.index.name = 'YEARS'
        
            # Save
            df_out.to_csv(os.path.join(datadir, 'processed', 'globalmean_annual', 
                           f"tas_annual_{esm}_{scenario}_{run}_{biascorr}_{yearstart}-{yearend}.csv"))
            
            # remove gridded files
            for filepath in annual_climate_filepaths:
                os.remove(filepath)

            