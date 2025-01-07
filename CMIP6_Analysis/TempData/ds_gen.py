import math
import numpy as np
import xarray as xr
import calendar
import pandas as pd
import datetime
import os
import cftime
import matplotlib.pyplot as plt
from scipy import stats
import sys
import seaborn as sns
from scipy.stats import linregress, pearsonr, percentileofscore, norm
sys.path.append('/home/users/chingosa/Functions/')
import CMIPFuncs as func
from multiprocessing import Pool, cpu_count
import time

def generatePDF_bins(da, nBins):
    return np.histogram_bin_edges(da.values.flatten()[~np.isnan(da.values.flatten())], bins=nBins)

def generatePDF(da, bin_edges):
    name = da.name
    binWidth = bin_edges[1]-bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers


    da_pdf = xr.apply_ufunc(
            lambda data: np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=bin_edges, density=True)[0], 
            da,
            input_core_dims=[['time']],
            output_core_dims=[[f"bins_{name}"]],
            kwargs={},
            dask="parallelized",
            dask_gufunc_kwargs={'allow_rechunk': True,
                               'output_sizes': {f"bins_{name}": len(bin_edges)-1}},
            output_dtypes=[float],
            vectorize=True,
        )
    
    da_pdf = da_pdf.assign_coords({f"bins_{name}": bin_centers})
    da_pdf = da_pdf.rename(f'{name}_pdf')
    return da_pdf

def estimateP(score, bin_centers, pdf, binWidth):
    pdf = pdf.flatten()
    cdf_approx = np.cumsum(pdf * binWidth)  # Cumulative sum to approximate the CDF

    # Find the index of the bin closest to the score
    index = np.argmin(np.abs(bin_centers - score))
    
    # Step 2: Identify the two bins to either side for interpolation
    if bin_centers[index] < score:
        left_idx, right_idx = index, index + 1
    else:
        left_idx, right_idx = index - 1, index
    
    # Ensure indices are within bounds
    left_idx = max(left_idx, 0)
    right_idx = min(right_idx, len(bin_centers) - 1)
    
    # Step 3: Linearly interpolate the percentile
    x_left, x_right = bin_centers[left_idx], bin_centers[right_idx]
    y_left, y_right = cdf_approx[left_idx], cdf_approx[right_idx]
    
    # Perform linear interpolation
    percentile_score = y_left + (y_right - y_left) * ((score - x_left) / (x_right - x_left))

    return percentile_score*100  # Return as a numpy array
    

def estimatePx(da_pdf, da_score):
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

def valueAtPx(da_values, da_px):
    da = xr.apply_ufunc(lambda x, y: np.nanpercentile(x,y),
                  da_values, 
                  da_px,
                  input_core_dims=[['time'],[]],
                  dask_gufunc_kwargs={
                      'allow_rechunk': True},
                  vectorize=True,
                  dask="parallelized",
                  output_dtypes=[float]
                 )
    return da

modelNames = pd.read_excel('/home/users/chingosa/CMIP6/CMIP6Models.xlsx')
model = modelNames.ModelName[0]

periods = ['historical', 'ssp245']
variables = ['tas', 'huss', 'ps']
percentiles = [0.001, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 99]


## Grab the variables
ds_hist, ds_ssp245 = func.dsRetrieve(variables, model, periods, gridNorm=True)

# calculate q_sat
ds_hist['q_sat'] = ((0.622*(0.6113 * np.exp(2500000/461*((1/273.15)-(1/ds_hist['tas']))))) / ((ds_hist['ps']/1000) - ((0.6113 * np.exp(2500000/461*((1/273.15)-(1/ds_hist['tas']))))*(1-0.622))))
ds_ssp245['q_sat'] = ((0.622*(0.6113 * np.exp(2500000/461*((1/273.15)-(1/ds_ssp245['tas']))))) / ((ds_ssp245['ps']/1000) - ((0.6113 * np.exp(2500000/461*((1/273.15)-(1/ds_ssp245['tas']))))*(1-0.622))))

## Calculate MSE
cp = 1004.6
Lv = 2500000

ds_hist['MSE'] = cp*ds_hist.tas + Lv*ds_hist.huss
ds_ssp245['MSE'] = cp*ds_ssp245.tas + Lv*ds_ssp245.huss

ds_hist['MSE_MOP_land'] = func.getMeansOverPercentile_difVar(ds_hist.tas.where(ds_hist.landseamask == 1), ds_hist.MSE.where(ds_hist.landseamask == 1), percentiles, avgOut = ['time'])
ds_ssp245['MSE_MOP_land'] = func.getMeansOverPercentile_difVar(ds_ssp245.tas.where(ds_hist.landseamask == 1), ds_ssp245.MSE.where(ds_hist.landseamask == 1), percentiles, avgOut = ['time'])

hist_bin_edges = generatePDF_bins(ds_hist.MSE.where(ds_hist.landseamask == 1), 500)  # this function generates the bin edges for the PDF generator - takes awhile
ds_hist['MSE_pdf'] = generatePDF(ds_hist.MSE.where(ds_hist.landseamask == 1), hist_bin_edges)
ds_hist.MSE_pdf.to_netcdf(f'hist_{model}_11_28.nc')

ssp245_bin_edges = generatePDF_bins(ds_ssp245.MSE.where(ds_ssp245.landseamask == 1), 500)  # this function generates the bin edges for the PDF generator - takes awhile
ds_ssp245['MSE_pdf'] = generatePDF(ds_ssp245.MSE.where(ds_ssp245.landseamask == 1), ssp245_bin_edges)
ds_ssp245.MSE_pdf.to_netcdf(f'ds_ssp245_{model}_11_28.nc')

del ds_hist, ds_ssp245



ds_hist, ds_ssp245 = func.dsRetrieve(variables, model, periods, gridNorm=True)

# calculate q_sat
ds_hist['q_sat'] = ((0.622*(0.6113 * np.exp(2500000/461*((1/273.15)-(1/ds_hist['tas']))))) / ((ds_hist['ps']/1000) - ((0.6113 * np.exp(2500000/461*((1/273.15)-(1/ds_hist['tas']))))*(1-0.622))))
ds_ssp245['q_sat'] = ((0.622*(0.6113 * np.exp(2500000/461*((1/273.15)-(1/ds_ssp245['tas']))))) / ((ds_ssp245['ps']/1000) - ((0.6113 * np.exp(2500000/461*((1/273.15)-(1/ds_ssp245['tas']))))*(1-0.622))))

## Calculate MSE
cp = 1004.6
Lv = 2500000

ds_hist['MSE'] = cp*ds_hist.tas + Lv*ds_hist.huss
ds_ssp245['MSE'] = cp*ds_ssp245.tas + Lv*ds_ssp245.huss

ds_hist['MSE_pdf'] = xr.open_dataset(f'hist_{model}_11_28.nc').MSE_pdf
ds_ssp245['MSE_pdf'] = xr.open_dataset(f'ds_ssp245_{model}_11_28.nc').MSE_pdf

### Land First
for var in ['tas','huss', 'q_sat', 'MSE']:
    ds_hist[f'{var}_MOP_land'] = func.getMeansOverPercentile_difVar(ds_hist.tas.where(ds_hist.landseamask == 1), ds_hist[var].where(ds_hist.landseamask == 1), percentiles, avgOut = ['time'])
    ds_ssp245[f'{var}_MOP_land'] = func.getMeansOverPercentile_difVar(ds_ssp245.tas.where(ds_ssp245.landseamask == 1), ds_ssp245[var].where(ds_hist.landseamask == 1), percentiles, avgOut = ['time'])


ds_hist['MSE_land_px']  = estimatePx(ds_hist.MSE_pdf, ds_hist.MSE_MOP_land)
ds_ssp245['MSE_land_px']  = estimatePx(ds_ssp245.MSE_pdf, ds_ssp245.MSE_MOP_land)

for var in ['tas', 'q_sat', 'huss']:
    ds_hist[f'{var}_ocean_px'] = valueAtPx(ds_hist[var].where(ds_hist.landseamask == 0), ds_hist.MSE_land_px.mean('lon'))
    ds_ssp245[f'{var}_ocean_px'] = valueAtPx(ds_ssp245[var].where(ds_ssp245.landseamask == 0), ds_hist.MSE_land_px.mean('lon'))



ds = xr.Dataset()

ds['landseamask'] = ds_hist.landseamask
ds['latWeight'] = ds_hist.latWeight
ds['landFracWeight'] = ds_hist.landFracWeight
ds['d_q_sat_land'] = (ds_ssp245.q_sat_MOP_land - ds_hist.q_sat_MOP_land)
ds['d_q_sat_ocean'] = (ds_ssp245.q_sat_ocean_px - ds_hist.q_sat_ocean_px).weighted(ds_hist.oceanFracWeight * ds_hist.latWeight).mean('lon')

ds['d_tas_land'] = (ds_ssp245.tas_MOP_land - ds_hist.tas_MOP_land)
ds['d_tas_ocean'] = (ds_ssp245.tas_ocean_px - ds_hist.tas_ocean_px).weighted(ds_hist.oceanFracWeight * ds_hist.latWeight).mean('lon')

ds['alpha_L'] = ((ds.d_q_sat_land/ds_hist.q_sat_MOP_land) / ds.d_tas_land)

ds['alpha_O'] = ((ds.d_q_sat_ocean/ds_hist.q_sat_ocean_px.weighted(ds_hist.oceanFracWeight * ds_hist.latWeight).mean('lon')) / ds.d_tas_ocean)

ds['gamma_top'] = (cp + (Lv*ds.alpha_O* ds_hist.huss_ocean_px.weighted(ds_hist.oceanFracWeight * ds_hist.latWeight).mean('lon')))

ds['gamma_bottom'] = (cp + (Lv*ds.alpha_L* ds_hist.huss_MOP_land))

ds['gamma'] = ds.gamma_top/ds.gamma_bottom 

ds['d_tas_land_predict'] = (ds.gamma*ds.d_tas_ocean)
ds['d_px'] = (ds_ssp245.MSE_land_px - ds_hist.MSE_land_px)
ds['dfracland_ocean'] = ds['d_tas_land']/ds['d_tas_ocean']
ds = ds.drop_vars(['d_q_sat_land', 'd_q_sat_ocean', 'alpha_L', 'alpha_O', 'gamma_top', 'gamma_bottom'])
ds = ds.expand_dims(model = [model])

ds.to_netcdf(f'{model}_11_22.nc')
