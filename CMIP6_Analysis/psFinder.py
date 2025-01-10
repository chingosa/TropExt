import sys
sys.path.append('/home/users/chingosa/TropExt/Functions/')
import CMIPFuncs as func
import numpy as np
import xarray as xr
import pandas as pd
import os
from pyesgf.search import SearchConnection
conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True) #German one
os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "on"

getPSmodel = ['EC-Earth3',
 'EC-Earth3-AerChem',
 'EC-Earth3-AerChem',
 'EC-Earth3-AerChem',
 'MIROC6',
 'MIROC6',
 'MPI-ESM1-2-LR',
 'MPI-ESM1-2-LR',
 'MPI-ESM1-2-LR',
 'MPI-ESM1-2-LR',
 'MRI-ESM2-0',
 'MRI-ESM2-0',
 'MRI-ESM2-0',
 'MRI-ESM2-0']
    
getPSperiod = ['piControl',
 'piControl',
 'historical',
 'ssp245',
 'historical',
 'ssp245',
 'abrupt-4xCO2',
 'piControl',
 'historical',
 'ssp245',
 'abrupt-4xCO2',
 'piControl',
 'historical',
 'ssp245']

def calc(i):
        
    getPSmodel = ['EC-Earth3',
 'EC-Earth3-AerChem',
 'EC-Earth3-AerChem',
 'EC-Earth3-AerChem',
 'MIROC6',
 'MIROC6',
 'MPI-ESM1-2-LR',
 'MPI-ESM1-2-LR',
 'MPI-ESM1-2-LR',
 'MPI-ESM1-2-LR',
 'MRI-ESM2-0',
 'MRI-ESM2-0',
 'MRI-ESM2-0',
 'MRI-ESM2-0']
        
    getPSperiod = ['piControl',
 'piControl',
 'historical',
 'ssp245',
 'historical',
 'ssp245',
 'abrupt-4xCO2',
 'piControl',
 'historical',
 'ssp245',
 'abrupt-4xCO2',
 'piControl',
 'historical',
 'ssp245']

    files = []
    model = getPSmodel[i]    
    period = getPSperiod[i]
    print(model, period)
    
    query = conn.new_context(project="CMIP6",     
                                         experiment_id=period,
                                         source_id = model,
                                         frequency = 'mon', 
                                         variable_id = 'ps')
    
       
    results = query.search()
    for i in range(len(results)):
        try:
            hit = results[i].file_context().search()
        except:
            hit = results[i].file_context().search()
        files += list(map(lambda f: {'model': model,
                                     'filename': f.filename, 
                                     'download_url': f.download_url, 
                                     'opendap_url': f.opendap_url}, hit))
    df = pd.DataFrame.from_dict(files)
    dfOld = pd.read_csv(f'TempData/Paths_{model}.csv', index = None)
    dfOld = dfOld[~((dfOld['period'] == period) & (dfOld['Var'] == 'ps'))&(dfOld['model'] == model)] # removes the daily ps
    df = pd.concat([df, dfOld], ignore_index=True)
    df.to_csv(f'TempData/Paths_{model}.csv', index=None)

func.parallel_execution(calc, np.arange(len(getPSperiod)), processes=8)