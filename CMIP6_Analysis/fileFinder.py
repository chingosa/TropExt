import sys
sys.path.append('/home/users/chingosa/TropExt/Functions/')
import CMIPFuncs as func
import numpy as np
import xarray as xr
import pandas as pd
import os
from pyesgf.search import SearchConnection
os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "on"

def extract_varientID(string):  # also Known as extract_8_chars
    start_index = string.find('_r') 
    end_index = string.find('_', start_index+1)
    return string[start_index+1:end_index]

def extract_SimNums(string):
    try:
        start_index = string.find('_r')
        end_index = string.find('i', start_index)
        r = int(string[start_index + 2:end_index])

        start_index = string.find('i', end_index)
        end_index = string.find('p', start_index)
        i = int(string[start_index + 1:end_index])

        start_index = string.find('p', end_index)
        end_index = string.find('f', start_index)
        p = int(string[start_index + 1:end_index])

        start_index = string.find('f', end_index)
        end_index = string.find('_', start_index)
        f = int(string[start_index + 1:end_index])
        
        return r, i, p, f
    except Exception as e:
        # Provide feedback for debugging
        print(f"Error processing string '{string}': {e}")
        return None, None, None, None
   
def getPeriod(filename):
    if 'abrupt-4xCO2' in filename: return 'abrupt-4xCO2'
    elif 'piControl' in filename: return 'piControl'
    elif 'historical' in filename: return 'historical'
    elif 'ssp245' in filename: return 'ssp245'
    
def whichVar(filename):
    if 'tas' in filename: return 'tas'
    elif 'huss' in filename: return 'huss'
    elif 'hurs' in filename: return 'hurs'
    elif 'ps' in filename: return 'ps'

def whichTimestep(filename):
    if 'day' in filename: return 'day'
    elif 'mon' in filename: return 'mon'
    else : return np.nan

def whichGrid(filename):
    varID = extract_varientID(filename)
    start_index = filename.find(varID) +len(varID)
    end_index = filename.find('_', start_index+1)
    return filename[start_index+1:end_index]

def extractDates(string):
    ncFind = string.find('.nc') 
    ncBack = string.rfind('-', 0, ncFind)
    dback = (ncFind-ncBack)-1
    stop = string[ncBack+1:ncFind]
    start = string[ncBack-dback:ncBack]
    try:
        return extractYear(start), extractYear(stop)
    except:
        print(string)
def extractYear(Date):
    return int(Date[:4])

# conn = SearchConnection('https://esgf.ceda.ac.uk/esg-search', distrib=True) #UK one
conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True) #German one
# conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True)  #US one




## This comes up with a list of agencies and models - the models list is what we will use to search
Agencies = os.listdir(f'/badc/cmip6/data/CMIP6/CMIP/')
Models = [os.listdir(f'/badc/cmip6/data/CMIP6/CMIP/{Agency}') for Agency in Agencies]
models = []
for sublist in Models:
    models.extend(sublist)

print(len(models))

def findFilePaths(model): 
    if os.path.isfile(f'/home/users/chingosa/TropExt/CMIP6_Analysis/TempData/Paths_{model}.csv'):
        print('skip')
        return None
    else:
        print(model) # for monitoring progress of parallel execution
        files = []
        for exid in ['abrupt-4xCO2','piControl','historical','ssp245']:
            for var in ['tas', 'huss', 'ps']:
                query = conn.new_context(project="CMIP6",     
                                         experiment_id=exid,
                                         source_id = model,
                                         frequency = 'day', 
                                         # member_id="r1i1p1f1",
                                         variable_id = var)

       
                results = query.search()
                for i in range(len(results)):
                    try:
                        hit = results[i].file_context().search()
                    except:
                        print(model, "don't work so good")
                        return
                        # hit = results[i].file_context().search()
                    files += list(map(lambda f: {'model': model,
                                                 'filename': f.filename, 
                                                 'download_url': f.download_url, 
                                                 'opendap_url': f.opendap_url}, hit))
                
                if (len(results) == 0) & (var == 'ps'):
                    query = conn.new_context(project="CMIP6",     
                                         experiment_id=exid,
                                         source_id = model,
                                         frequency = 'mon', 
                                         # member_id="r1i1p1f1",
                                         variable_id = var)

       
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
        if len(df) == 0: return None
        df = df.dropna() # some opendap_urls are not found - so get rid of those
        df[['r', 'i', 'p', 'f']] = pd.DataFrame(
            df['filename'].apply(extract_SimNums).tolist(),
            index=df.index
        )
                
        df['Varient'] = df.filename.apply(extract_varientID)
        df['period'] = df.filename.apply(getPeriod)
        df['Var'] = df.filename.apply(whichVar)
        df['grid'] = df.filename.apply(whichGrid)
        df['timeStep'] = df.filename.apply(whichTimestep)
        
        # figure out the time each models cover
        df[['start', 'stop']] = df['filename'].apply(extractDates).apply(pd.Series)
        
        df.to_csv(f'/home/users/chingosa/TropExt/CMIP6_Analysis/TempData/Paths_{model}.csv')   # Saves to CSV
        return None

func.parallel_execution(findFilePaths, models, processes=8)