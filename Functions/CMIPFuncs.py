def getVariable(variable, modelName, simulations):
    import xarray as xr
    import numpy as np
    import calendar
    import pandas as pd
    import datetime
    import os
    import cftime
    modelNames = pd.read_excel('/home/users/chingosa/CMIP6/CMIP6Models.xlsx')
    i = modelNames.index[modelNames['ModelName'] == modelName].tolist()[0]
    org = modelNames.ModelInstitution[i]
    model = modelNames.ModelName[i]
    fCode = modelNames.f[i]
    grid = modelNames.grid[i]
    ensambleMember = 1
    period = 'day'
    if variable == 'ps': period = 'Amon' #6hrLev
    if variable == 'sftof': period = 'Ofx'

    #File Path for historical and ssp all daily variable
    if simulations == 'ssp245':
        folder_path = f'/badc/cmip6/data/CMIP6/ScenarioMIP/{org}/{model}/ssp245/r{ensambleMember}i1p1f{fCode}/{period}/{variable}/{grid}/files/'     
    elif simulations =='historical':
        
        if model in ['BCC-CSM2-MR'] : ensambleMember += 1 # forwhatever reason there is no 1st ensamble member for this one
        folder_path = f'/badc/cmip6/data/CMIP6/CMIP/{org}/{model}/historical/r{ensambleMember}i1p1f{fCode}/{period}/{variable}/{grid}/files/'
    
    if len(os.listdir(folder_path)) == 1:
        folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    else:
        folder_path = os.path.join(folder_path, 'latest')
            
    # List and print all file names in the specified folder
    file_paths = [
    os.path.join(folder_path, file_name) 
    for file_name in os.listdir(folder_path) 
    if os.path.isfile(os.path.join(folder_path, file_name))]


    file_paths = [x for x in file_paths if x != '/badc/cmip6/data/CMIP6/CMIP/NCAR/CESM2-WACCM/historical/r1i1p1f1/day/psl/gn/files/d20190227/psl_day_CESM2-WACCM_historical_r1i1p1f1_gn_18500101-20150101.nc']
    return file_paths

def dsRetrieve(variables, model, periods, gridNorm = False):
    import xarray as xr
    import numpy as np
    import calendar
    import pandas as pd
    import datetime
    import os
    import cftime
    import CMIPFuncs as func

    ds_hist, ds_ssp245 = None, None
    for period in periods:
        for var in variables:
            ds_merge = xr.open_mfdataset(func.getVariable(var, model, period))
            
            if var == variables[0]: ds = ds_merge
            else: ds = xr.merge([ds, ds_merge[var]])
        del ds_merge   
        
        # Cut down region and add Land Mask
        if gridNorm:
            normGrid = xr.open_dataset('/badc/cmip6/data/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-CM4/ssp245/r1i1p1f1/fx/sftlf/gr1/latest/sftlf_fx_GFDL-CM4_ssp245_r1i1p1f1_gr1.nc').sftlf
            ds = ds.interp_like(normGrid, kwargs={"fill_value": "extrapolate"})

        ds = func.regionTimeCut(ds, period) 
        ds = func.addLandMask(ds)
    
        if 'ps' in variables:
            ds['ps'] = ds['ps'].chunk({'time': -1})
            ds['ps'] = ds['ps'].interpolate_na(dim='time', method='linear', fill_value='extrapolate')
            ds['ps'] = ds['ps'].chunk({'time': 1})
        ds['latWeight'] = np.cos(np.deg2rad(ds.lat))
        ds['landFracWeight'] = ds.landseamask.mean('lon')
        ds['oceanFracWeight'] = 1 - ds['landFracWeight']

        
        if    period == 'ssp245': 
            ds_ssp245 = ds
        elif  period == 'historical': 
            ds_hist   = ds
    del ds
    return ds_hist, ds_ssp245


def regionTimeCut(ds, period, latchoice = [-20,20]):
    import xarray as xr
    import numpy as np
    import calendar
    import pandas as pd
    import datetime
    import os
    import cftime
    if period == 'historical':
        start_date = cftime.DatetimeNoLeap(1980, 1, 1)
        end_date = cftime.DatetimeNoLeap(2000, 12, 31)
    elif period == 'ssp245':
        start_date = cftime.DatetimeNoLeap(2080, 1, 1)
        end_date = cftime.DatetimeNoLeap(2100, 12, 31)

    if ds.time.dtype == '<M8[ns]':
        start_date, end_date  = np.datetime64(start_date), np.datetime64(end_date)
    elif type(ds.time.values[0]) == cftime._cftime.Datetime360Day:
        if period == 'historical':
            start_date = cftime.Datetime360Day(1980, 1, 1)
            end_date = cftime.Datetime360Day(2000, 12, 30)
        elif period == 'ssp245':
            start_date = cftime.Datetime360Day(2080, 1, 1)
            end_date = cftime.Datetime360Day(2100, 12, 30)
            
    tropical_region = ds.sel(lat=slice(latchoice[0], latchoice[1]))
    # tropical_region = ds.sel(lat=slice(-10,10))  #delete these
    # tropical_region = tropical_region.sel(lon=slice(-10,10)) #delete these
    tropical_region = tropical_region.sel(time=slice(start_date, end_date))
    return tropical_region


def addLandMask(ds):
    import xarray as xr
    import numpy as np
    import calendar
    import pandas as pd
    import datetime
    import os
    import cftime
    '''
    Takes in a dataset finds the variable that his lat lon coords and makes a 2D data array with the coursened version of the land mask
    Adds that back into the ds
    It defaults to using NOAAs GFDL-CM4 sftlf land mask and then iterpolating 
    We choose to use this coarser version and interpolate to different grids bc it is a similar resolution used in all simulations
    '''
    landMask = xr.open_dataset('/badc/cmip6/data/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-CM4/ssp245/r1i1p1f1/fx/sftlf/gr1/latest/sftlf_fx_GFDL-CM4_ssp245_r1i1p1f1_gr1.nc').sftlf
    vars = list(ds.keys())
    
    for i in np.arange(len(vars)):
        if set(['lat', 'lon']).issubset(list(ds[vars[i]].coords)):
            chosenVar = list(ds.keys())[i]
    
    da = ds[chosenVar].isel(time=0) # this line would be an issue if there isn't time variable
    coarsened_da = landMask.interp_like(da, kwargs={"fill_value": "extrapolate"})
    coarsened_da = (coarsened_da/100)
    coarsened_da = xr.apply_ufunc(np.round, coarsened_da)
    ds['landseamask'] = coarsened_da
    return ds

def mean_above_percentiles(data, percentiles, axis):
    import xarray as xr
    import numpy as np
    # Calculate the threshold for each percentile
    thresholds = np.nanpercentile(data, percentiles, axis = axis )
    means = np.full_like(percentiles, np.nan, dtype=np.float64)  # Pre-allocate array for means

    # Calculate mean for values above each threshold
    for i, threshold in enumerate(thresholds):
        above_threshold = data[data > threshold]
        if above_threshold.size > 0:
            means[i] = np.nanmean(above_threshold)
            
    return means

def mean_above_percentiles_difVar(data, dataCon, percentiles, axis):
    import xarray as xr
    import numpy as np
    # Calculate the threshold for each percentile
    thresholds = np.nanpercentile(data, percentiles, axis=axis)
    means = np.full_like(percentiles, np.nan, dtype=np.float64)  # Pre-allocate array for means

    # Calculate mean for values above each threshold
    for i, threshold in enumerate(thresholds):
        above_threshold = dataCon[data > threshold]
        if above_threshold.size > 0:
            means[i] = np.nanmean(above_threshold)
            
    return means
    
def getMeansOverPercentile(da, percentiles, avgOut = ['time']):
    import xarray as xr
    import numpy as np
    import CMIPFuncs as func
    '''
    This function takes a dataset ds and a specific variable in it and for a list of percentiles finds the mean value of the data over a given percentile MOP for short 
    It defaults to taking in any ds and then reducing by time ie maintaining lat lon, and lev but if you wanted to get rid of those too you can change the average out function
    '''
    import xarray as xr
    import numpy as np
    import CMIPFuncs as func
    axis = np.arange(len(avgOut))
    da_new = xr.apply_ufunc(
    lambda x: mean_above_percentiles(x, percentiles, axis),  #add back func
        da,
        input_core_dims=[avgOut],
        output_core_dims=[["MOP"]],
        dask_gufunc_kwargs={
            'output_sizes': {"MOP": len(percentiles)},
            'allow_rechunk': True
        },
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float]
    )
    da_new = da_new.assign_coords(MOP=percentiles)
    return da_new

def getMeansOverPercentile_difVar(da, da_con, percentiles, avgOut = ['time']):
    '''
        the first specified variable is what it will use to find the percentile of and the second is the conVar which is the variable that we want the mean of 
    '''
    import xarray as xr
    import numpy as np
    import CMIPFuncs as func
    
    axis = np.arange(len(avgOut))
    
    da_new = xr.apply_ufunc(
    lambda x,y: mean_above_percentiles_difVar(x, y, percentiles, axis),  #add back func
    da, da_con,
    input_core_dims=[avgOut,avgOut],
    output_core_dims=[["MOP"]],
    dask_gufunc_kwargs={
        'output_sizes': {"MOP": len(percentiles)},
        'allow_rechunk': True
    },
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float]
    )
    da_new = da_new.assign_coords(MOP=percentiles)
    return da_new



def getPercentile(ds, var, percentiles, avgOut = ['time']):
    '''
    This function takes a dataset ds and a specific variable in it and for a list of percentiles finds the percentile 
    It defaults to taking in any ds and then reducing by time ie maintaining lat lon, and lev but if you wanted to get rid of those too you can change the average out function
    '''
    import xarray as xr
    import numpy as np
    import CMIPFuncs as func
    ds[f'{var}_percentile'] = xr.apply_ufunc(
    lambda x: np.percentile(x, percentiles),
    ds[var],
    input_core_dims=[avgOut],
    output_core_dims=[["percentile"]],
    dask_gufunc_kwargs={
        'output_sizes': {"percentile": len(percentiles)},
        'allow_rechunk': True
    },
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float]
    )
    ds[f'{var}_percentile'] = ds[f'{var}_percentile'].assign_coords(percentile=percentiles)
    return ds


def parallel_execution(func, inputs, processes=None):
    from multiprocessing import Pool, cpu_count

    # Set the number of processes to use (default: number of CPU cores)
    if processes is None:
        processes = cpu_count()

    # Create a pool of worker processes
    with Pool(processes=processes) as pool:
        # Map the function to inputs and distribute across processors
        results = pool.map(func, inputs)

    return results

def find_files_by_name(directory, search_string):
    #straight outta chatGPT
    matching_files = []
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if search_string in file:
                file_path = os.path.join(root, file)
                matching_files.append(file_path)
    
    return matching_files


def testFunc(k):
    import numpy as np
    return np.arange(0,k,0.5)
    