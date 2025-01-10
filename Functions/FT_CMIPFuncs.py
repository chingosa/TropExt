import requests
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr  
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats 
import os
import cftime

def grabDS(model, period, Vars):
    try:
        return xr.open_mfdataset([f'TempData/{model}_{period}_{Var}_processed.nc' for Var in Vars], use_cftime = True)
    except:
        print('No File Found')

def AddLandMask(ds):
    landMask = xr.open_dataset('/badc/cmip6/data/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-CM4/ssp245/r1i1p1f1/fx/sftlf/gr1/latest/sftlf_fx_GFDL-CM4_ssp245_r1i1p1f1_gr1.nc').sftlf
    ds['landseamask'] = (landMask/100).round()
    return ds

def CalcSatSpecificHumidity(ds):
    Rv = 461 # J·K–1·kg–1
    Lv = 2.5E6 # J·kg–1
    eo = 611.3 # Pa
    To = 273.15 # K
    epi = 0.622 # ratio between vapor constant for dry air and water vapor
    satSpecificVP = eo*np.exp((Lv/Rv)*((1/To)-(1/ds.tas)))

    ds['qsat'] = (epi*satSpecificVP)/(ds.ps - satSpecificVP*(1-epi))
    return ds

def CalcMSE(ds):
    cp = 1004.6 #  J kg−1 K−1
    Lv = 2.5E6 # J·kg–1
    ds['MSE'] = cp*ds.tas + Lv*ds.huss
    return ds
    

def meanOverPercentile(ds, Conditioner, Vars, quantiles, dimOut, additionalMask = True, add_brev= ''):
    ds[f'{Conditioner}_qant'] = ds[Conditioner].chunk({'time': None}).quantile(quantiles, dimOut)
    
    mean_values = []
    # Iterate over quantiles and calculate the mean where tas > quantile
    for q in quantiles:
        # Mask where tas > quantile
        mask = (ds[Conditioner] > ds.tas_qant.sel(quantile=q)) & additionalMask
        # Compute the mean of tas along time for masked values
        for Var in Vars:
            mean_Var = ds[Var].where(mask).mean(dim=dimOut)
            mean_values.append(mean_Var)
    for i, Var in enumerate(Vars):
        ds[f'{Var}_MOP{add_brev}'] = xr.concat(mean_values[i::len(Vars)], dim = 'quantile')

    return ds

def generatePDF_bins(da, nBins):
    '''np.histogram_bin_edges except it filters out nanvalues first bc that sort of breaks it I guess'''
    return np.histogram_bin_edges(da.values.flatten()[~np.isnan(da.values.flatten())], bins=nBins)

def generatePDF(da, bin_edges, dimOver = ['time']):
    # Compute bin centers
    name = da.name
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Define the function to compute histogram with density normalization
    def compute_histogram(data):
        # data = data.flatten()
        if data.size == 0:
            return np.zeros(len(bin_edges) - 1)
        hist, _ = np.histogram(data, bins=bin_edges, density=True)
        return hist

    # Apply the histogram function along the 'time' dimension
    da_pdf = xr.apply_ufunc(
                            compute_histogram,
                            da,
                            input_core_dims=[dimOver],
                            output_core_dims=[[f"bins_{name}"]],
                            vectorize=True,
                            dask='parallelized',
                            dask_gufunc_kwargs={'output_sizes': {f"bins_{name}": len(bin_edges)-1},
                                               'allow_rechunk': True},
                            output_dtypes=[float]
                            )

    # Assign coordinates and rename for clarity
    da_pdf = da_pdf.assign_coords({f"bins_{name}": bin_centers})
    da_pdf = da_pdf.rename(f'{da.name}_pdf')

    return da_pdf

def estimateP(score, bin_centers, pdf, binWidth):
    '''
    Description:
    this function is a helper function to estimatePx - it estimates a value of percentile for a given score of a variable from a pdf of that variable using linear interpolation across a cdf
    Parameters
    ----------
    score - the value you want to estimate
    bin_centers - the center of each bin the pdf is applied over
    pdf - the probability density function describing the distribition of a variable
    binwidth - the width of bins idk

    returns
    -------
    percentile_score - a value 0-100 of the estimated percentile that the prodived score corresponds to

    '''
     
    pdf = pdf.flatten()
    cdf_approx = np.cumsum(pdf * binWidth)  # Cumulative sum to approximate the CDF
    percentile_score = np.interp(score, bin_centers, cdf_approx)
    return percentile_score  # Return as a numpy array
    

def estimatePx(da_pdf, da_score):
    '''
    Description:
    Estimates the percentile of a given score given pdfs and scores
    Parameters
    ----------
    da_pdf - dataarray with a probability denisity function running along axis bins_name
    da_score - dataarray with scores of name variable that you want to get the estimated percentiles of
    returns
    -------
    da_px - dataarray with same dimensions as da_score
    '''
     
    name = da_pdf.name[:-4]
    bin_centers= da_pdf[f'bins_{name}'].values
    binWidth = bin_centers[1]-bin_centers[0]
    
    da_px = xr.apply_ufunc(
        lambda x, y: estimateP(y, bin_centers, x, binWidth), 
        da_pdf,  
        da_score,  
        input_core_dims=[[f"bins_{name}"], []], 
        output_core_dims=[[]],  
        vectorize=True,  
        dask="parallelized",  
        dask_gufunc_kwargs={'allow_rechunk': True},  
        output_dtypes=[float]  
    )
    return da_px

def estimateValueFromPDF(percentile, bin_centers, pdf, binWidth):
    '''
    Description:
    Helper function for ValuefromPx() - used to interpolate the value from a percentile using a cumulative distribution function and a percentile
    Note that percentile is provided in 0-100 percentile space and devided to work with a 0-1 pdf
    Parameters
    ----------
    percentile: percentile - 0-100 
    bin_centers - 1D array - essentially the values from bins_name
    pdf - density at bin centered at bin_centers
    binWidth - width of bin - single value (maybe make list support later)    
    
    returns
    -------
    float - of interpolated value for a percentile

    '''
    pdf = pdf.flatten()
    cdf_approx = np.cumsum(pdf * binWidth)  # Cumulative sum to approximate the CDF
    return np.interp(percentile, cdf_approx, bin_centers)
    
def ValuefromPx(da_pdf, da_px, missingDim = [], latWise = False):
    '''
    Description:
    This function estimates the value at xth percentile

    Parameters
    ----------
    da_pdf: dataarray with ['lat', 'lon', f'bins_{name}'] - bins_name are the bincenters of the probability density function
            the sum of the values aligned along bins_name must equal one

    da_px: dataarray with ['lat', 'lon', 'MOP'] - specifies the percentile in 0-100 space (devision happens within estimateValueFromPDF
    
    returns
    -------
    da_px: data array with same output dimensions as da_px (input) - its an estimate from the pdf of the value of var at xth percentile

    '''
     
    name = da_pdf.name[:-4]
    bin_centers= da_pdf[f'bins_{name}'].values
    binWidth = bin_centers[1]-bin_centers[0]
    if latWise: outCores = ["quantile"]
    else: outCores = ["quantile" ,'lon']
    da_px = xr.apply_ufunc(
        lambda x, y: estimateValueFromPDF(y, bin_centers, x, binWidth), 
        da_pdf,  # usually has lat, lon, and bins
        da_px,  # usually has lat lon and MOP
        input_core_dims=[[f"bins_{name}"], outCores], 
        output_core_dims=[outCores],  
        vectorize=True,  
        dask="parallelized",  
        dask_gufunc_kwargs={'allow_rechunk': True},  
        output_dtypes=[float]  
    )
    return da_px

def download(url, filename):
    """
    Stolen from https://utcdw.physics.utoronto.ca/UTCDW_Guidebook/Chapter3/section3.4_climate_model_output_from_ESGF.html#downloading-data

    Just a downloading script 
    Supply the dowload URLS and the filename to save to - they save directly to the local environment - would need to change this if you want to be more organizied
    """

    print("Downloading ", filename)
    r = requests.get(url, stream=True)
    total_size, block_size = int(r.headers.get('content-length', 0)), 1024
    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=total_size//block_size,
                         unit='KiB', unit_scale=True):
            f.write(data)

    if total_size != 0 and os.path.getsize(filename) != total_size:
        print("Downloaded size does not match expected size!\n",
              "FYI, the status code was ", r.status_code)


## This is the final time cropping functionality - 6 lines that took 6 days to figure out
def doTimeCrop(ds, model, period, span_df, filePaths):
    '''
    Grabs 20 years of time using cftime functionality (hopefully this works for all model)
    the start and end of these are agreed on using the span_df which is calculated in 10_12_24

    roughly 7300 points per ds depending on time dtype
    '''

    
    mask = (span_df.model == model) & (span_df.period == period)
    end = int(span_df.loc[mask, 'stopShared'].iloc[0])  # Extract the scalar value safely
    mask = ((ds['time.year'] > end-20) & (ds['time.year'] <= end))
    ds = ds.sel(time=mask)
    ds = ds.sel(time=~ds['time'].to_index().duplicated())
    return ds



def processModel(model, experiment, reducedFPs, spanDF_FP):
    '''
    
    Model Preprocessing for CMIP6 4xCO2 and PiControl or SSP245 and Hist

    Grid norm is what everything is interpolated to 
    File Paths are the reduced filepaths of just the times we are interested in 
    span_df - describes the temporal coverage of models and periods - so we can select a shared 20 year period

    For Each period and variable of the provided model we download the files
    then find the 20 years of interest, then interpolate to an agreed upon grid based on the gridding of GFDL-CM4
    then we crop to +- 40 N/S to look at the tropics
    If we are looking at 'ps' which is a monthly variable we need to pull in another ds (tas in this case) to interpolate time to

    We try and rechunk but I'm pretty sure that doesn't get saved lol
    We save it and then delete the source files bc they are alot larger

    This takes ~10 mins per model
    '''
    
    normGrid = xr.open_dataset('/badc/cmip6/data/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-CM4/ssp245/r1i1p1f1/fx/sftlf/gr1/latest/sftlf_fx_GFDL-CM4_ssp245_r1i1p1f1_gr1.nc').sftlf
    filePaths = pd.read_csv(reducedFPs, index_col = None)
    span_df = pd.read_csv(spanDF_FP, index_col=None)

    filePaths = filePaths[(filePaths.model == model) & (filePaths.experiment == experiment)].reset_index(drop=True)
        
        
    for period in filePaths.period.unique():
        for Var in ['tas', 'huss', 'ps']:  #'hurs'

            saveName = f'TempData/{model}_{period}_{Var}_processed.nc'
            paths = filePaths[(filePaths['period'] == period) & (filePaths['Var'] == Var)].reset_index(drop = True)
            if not os.path.exists(saveName): # Check if processed dataset has already been created
                for i in np.arange(len(paths)):   # if it hasn't go through each path in paths

                    if not os.path.exists(paths.filename[i]): # if that hasn't already been downloaded dowloadd it
                        url = paths.download_url[i]
                        filename = paths.filename[i]

                        download(url, f'TempData/{filename}')
            
            
                ds = xr.open_mfdataset([f'TempData/{fn}' for fn in paths.filename], combine='nested', concat_dim='time', use_cftime=True)
                ds = ds.drop_vars(['time_bnds'], errors = 'ignore')
                ds = doTimeCrop(ds, model, period, span_df, filePaths)
                ds = ds[Var]
                ds = ds.interp_like(normGrid, kwargs={"fill_value": "extrapolate"}).sel(lat = slice(-40,40))
                

                if Var == 'ps':
                    ds_tas = xr.open_dataset(f'TempData/{model}_{period}_tas_processed.nc', use_cftime=True)
                    ds = ds.interp_like(ds_tas, kwargs={"fill_value": "extrapolate"})
                    
                
                ds = ds.chunk({'time': -1, 'lat': 5})
                
                write_job = ds.to_netcdf(saveName, compute=False)
                with ProgressBar():
                    print(f"Writing to {saveName}")
                    write_job.compute()
                
            for k in [f'TempData/{fn}' for fn in paths.filename]:
                if os.path.exists(k): os.remove(k)

def ProcessStage2(model):
    for period  in ['piControl', 'abrupt-4xCO2']: #'abrupt-4xCO2'
        Vars = ['ps', 'tas', 'huss']
        ds = grabDS(model, period, Vars)
        ds = AddLandMask(ds)
        ds = ds.expand_dims(model = [model])
        ds = ds.expand_dims(period = [period])
        ds = CalcSatSpecificHumidity(ds)
        ds = CalcMSE(ds)
        
        quantiles = np.array([0.001, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 99])/100
        ds = ds.chunk({"time": -1})
        ds = meanOverPercentile(ds, 'tas', ['tas', 'huss', 'qsat', 'MSE'], quantiles, ['time'], additionalMask = (ds.landseamask == 1), add_brev = '_land')
        
        
        
        MSE_bin_edges = generatePDF_bins(ds.MSE, 500)
        
        ds[f'MSE_pdf'] = generatePDF(ds.MSE, MSE_bin_edges, dimOver = ['time'])
        ds['MSE_land_px']  = estimatePx(ds.MSE_pdf, ds.MSE_MOP_land)
            
            
        for Var in ['tas', 'huss', 'qsat']:
            if period == 'abrupt-4xCO2':
                piControl_land_px = xr.open_dataset(f'TempData/{model}_piControl_19_12_24.nc').MSE_land_px
                piControl_land_px = piControl_land_px.assign_coords({"period": ['abrupt-4xCO2']})
                px = piControl_land_px
            else: 
                px = ds.MSE_land_px
                
            bin_edges = generatePDF_bins(ds[Var], 500)
            ds[f'{Var}_pdf'] = generatePDF(ds[Var].where(ds.landseamask == 0), bin_edges, dimOver = ['time', 'lon'])
            ds[f'{Var}_ocean'] = ValuefromPx(ds[f'{Var}_pdf'], px) # tas_pdf is lat, and bins       ds_MSE_land_px is lat lon and quantile over land only
            ds[f'{Var}_land_mean'] = ds[Var].where(ds.landseamask == 1).mean('time')
            
        ds = ds.drop_vars([
         'time',
         'height',
         'huss',
         'ps',
         'tas',
         'qsat',
         'MSE',
         'MSE_pdf',
         'MSE_bins'
         'tas_qant',
         'bins_tas',
         'tas_pdf',
         'bins_huss',
         'huss_pdf',
         'bins_qsat',
         'qsat_pdf'], errors = 'ignore')
        
        ds.to_netcdf(f'TempData/{model}_{period}_19_12_24.nc')
        for Var in ['tas', 'huss', 'ps']: 
            if os.path.exists(f'TempData/{model}_{period}_{Var}_processed.nc'): 
                os.remove(f'TempData/{model}_{period}_{Var}_processed.nc')
        
def ProcessStage3(model):
    ds_p = xr.open_dataset(f'TempData/{model}_piControl_19_12_24.nc').squeeze('period')
    ds_4 = xr.open_dataset(f'TempData/{model}_abrupt-4xCO2_19_12_24.nc').squeeze('period')
    
    
    
    ## psudo RH
    ds_p['RH_x_land']     = ds_p.huss_MOP_land / ds_p.qsat_MOP_land
    ds_4['RH_x_land']     = ds_4.huss_MOP_land / ds_4.qsat_MOP_land
    
    ds_p['RH_land_mean']  = ds_p.huss_land_mean / ds_p.qsat_land_mean
    ds_4['RH_land_mean']  = ds_4.huss_land_mean / ds_4.qsat_land_mean
    
    
    ds_p['RH_ocean']     = ds_p.huss_ocean / ds_p.qsat_ocean
    ds_4['RH_ocean']     = ds_4.huss_ocean / ds_4.qsat_ocean
    
    ds = xr.Dataset()
    
    ds['dMSE_land_px'] =  ds_4.MSE_land_px - ds_p.MSE_land_px
    
    ## Tas
    ds['dtas_x_land']   = ds_4.tas_MOP_land - ds_p.tas_MOP_land
    ds['dtas_ocean']    = ds_4.tas_ocean - ds_p.tas_ocean
      
    ## Huss  
    ds['dhuss_x_land']  = ds_4.huss_MOP_land - ds_p.huss_MOP_land
    ds['dhuss_ocean']   = ds_4.huss_ocean - ds_p.huss_ocean
      
    ds['huss_x_land']   = ds_p.huss_MOP_land
    ds['huss_ocean']    = ds_p.huss_ocean
      
    ## qsat  
    ds['dqsat_x_land']  = ds_4.qsat_MOP_land - ds_p.qsat_MOP_land
    ds['dqsat_ocean']   = ds_4.qsat_ocean - ds_p.qsat_ocean
      
    ds['qsat_x_land']   = ds_p.qsat_MOP_land
    ds['qsat_ocean']    = ds_p.qsat_ocean
    ds['qsat_land_mean']= ds_p.qsat_land_mean
    
    ## Pusdo RH
    ds['dRH_x_land']    = ds_4.RH_x_land - ds_p.RH_x_land
    ds['dRH_ocean']     = ds_4.RH_ocean - ds_p.RH_ocean
    ds['dRH_land_mean'] = ds_4.RH_land_mean - ds_p.RH_land_mean
    
    # Alphas
    ds['alpha_land']    = (ds.dhuss_x_land/ds.huss_x_land) / ds.dtas_x_land
    ds['alpha_ocean']   = (ds.dhuss_ocean/ds.huss_ocean) / ds.dtas_ocean
    
    # epsilon and eta
    cp = 1004.6        #  J kg−1 K−1
    Lv = 2.5E6         #  J·kg–1
    
    ds['epsilon']      = (Lv*ds.alpha_land*ds.qsat_x_land)/(cp+Lv*ds.alpha_land*ds.huss_x_land)
    ds['eta']          = (ds.qsat_land_mean/ds.qsat_x_land)*(ds.epsilon / ds.alpha_land)
    
    ## Sensitivity Parameters
    ds['gamma_To']     = (cp+Lv*ds.alpha_ocean*ds.huss_ocean)/(cp+Lv*ds.alpha_land*ds.dhuss_x_land)
    ds['gamma_ro']     = (Lv*ds.qsat_ocean)/(cp+Lv*ds.alpha_land*ds.huss_x_land)
    
    ## dTLx Estimate
    ds['dtas_x_land_FT'] = ((1/(1+ds.epsilon*ds.dRH_x_land))*((ds.gamma_To * ds.dtas_ocean) + (ds.gamma_ro * ds.dRH_ocean) + 
                                                             ((ds.epsilon / ds.alpha_land) * (ds.dRH_x_land - ds.dRH_land_mean * ds.qsat_land_mean / ds.qsat_x_land))) -
                                                               - ((1/(1+ds.epsilon*ds.dRH_x_land))*(ds.epsilon*ds.alpha_land)*ds.dRH_x_land))
    
    ds['dTo_comp'] = (ds.gamma_To * ds.dtas_ocean)
    ds['dro_comp'] = (ds.gamma_ro * ds.dRH_ocean)
    ds['DRH_comp'] = ((ds.epsilon / ds.alpha_land) * (ds.dRH_x_land - ds.dRH_land_mean * ds.qsat_land_mean / ds.qsat_x_land))
    ds['dRHl_comp'] =  -((1/(1+ds.epsilon*ds.dRH_x_land))*(ds.epsilon*ds.alpha_land)*ds.dRH_x_land)
    ds.to_netcdf(f'{model}_FT_19_12_24.nc')
    for period in ['piControl', 'abrupt-4xCO2']:
        if os.path.exists(f'TempData/{model}_{period}_19_12_24.nc'): os.remove(f'TempData/{model}_{period}_19_12_24.nc')
    return None