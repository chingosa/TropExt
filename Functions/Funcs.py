def chime():
    import numpy as np
    from IPython.display import Audio, display, clear_output
    import time

    display(Audio(np.sin(2 * np.pi * 1 * np.linspace(0, 2, 4410 * 2)) + np.sin(2 * np.pi * 400 * np.linspace(0, 2, 4410 * 2)), rate=4410, autoplay=True))
    time.sleep(2)
    clear_output(wait=True)

def getRez(model):
    CRM_List = [ "CM1", "dam", "ICON_LEM_CRM", "ICON_NWP_CRM", "MESONH", "SAM_CRM", "SCALE", "UCLA-CRM", "UKMOi-vn11.0-CASIM", "UKMOi-vn11.0-RA1-T", "UKMOi-vn11.0-RA1-T-nocloud", "WRF_COL_CRM"]
    GCM_List = [ "WRF_GCM", "UKMO-GA7.1", "SPX-CAM", "SP-CAM", "SAM0-UNICON", "IPSL-CM6", "ICON_GCM", "GEOS_GCM", "ECHAM6_GCM", "CNRM-CM6-1", "CAM6_GCM", "CAM5_GCM"]  

    if model in CRM_List:
        rez = "CRM"
    elif model in GCM_List:
        rez = "GCM"
    else:
        print(f"{model} not it either list")
    return rez

def getArray(var, temp, model):
    import xarray as xr  
    import Funcs
    rez = Funcs.getRez(model)    
    filePath = f"/home/users/chingosa/RCEMIP/FinalAnalysis/AllModelData/{rez}_{model}/{temp}/{var}.nc"
    da = xr.open_dataarray(filePath)
    return da

def tree(directory='.', indent=''):
    import os
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            print(indent + f'[{item}]')
            tree(item_path, indent + '---')
        else:
            print(indent + item)

def RcumSum(R_dat, tempI, rez, val):
    R_dat = R_dat[((R_dat.temp == tempI) & (R_dat.rez == rez))]
    result_mean = R_dat.groupby('bins')['pdf'].mean().reset_index()
    result = pd.DataFrame({'bins': result_mean.bins,
                           'meanpdf' : result_mean['pdf'],
                          })
    result['cSum'] = result['meanpdf'].cumsum()
    
    cil = np.argmax(result['cSum'] >= val)
    ciu = np.argmax(result['cSum'] >= 1 - val)
    pdf_05 = np.interp(val, result['cSum'].iloc[cil-1:cil+1], result['bins'].iloc[cil-1:cil+1])
    pdf_95 = np.interp(1-val, result['cSum'].iloc[ciu-1:ciu+1], result['bins'].iloc[ciu-1:ciu+1])
    return pdf_05, pdf_95

def significance_stars(p_value):
    if p_value < 0.001:
        return '***'  # Significant at 99.9%
    elif p_value < 0.01:
        return '**'   # Significant at 99%
    elif p_value < 0.05:
        return '*'    # Significant at 95%
    else:
        return ''     # Not significant

def bold(p_value, sigVal):
    if p_value <= sigVal:
        return 'bold'
    else:
        return 'normal'

def do_r2(l1, l2, sigVal = 0.05):
    from scipy.stats import pearsonr
    correlation_coefficient, p_value = pearsonr(l1, l2)
    dict = {"r2" : round(correlation_coefficient**2, 2), 'stars' : significance_stars(p_value), 'bold': bold(p_value, sigVal)}
    return dict

def croppin(da, dims, red, style = 'Wing'):
    import numpy as np
    import xarray as xr
    import wrf

    '''
    This function takes a data array and reduces it along specified dimensions by a reduction factor following one of two procedures
    It returns an averaged data array with the same attributes as the input and a data array with a weighted array thats a sum of all the grid boxes that went into it
    '''
   
    da_weights = xr.ones_like(da)
    
    for red_dim, reduction in zip(dims, red):
            #Making the distrinctive grouping along the chosen axis
        groups = da[red_dim].size // reduction  #number of groups
        add = da[red_dim].size % reduction      #how much to add to the last column
        c = add//groups
        d = add%groups
        
        if style == 'Wing':
            group_sizes = np.concatenate([np.full(d, reduction+c+1), np.full(groups-d, reduction+c)])
        elif style == 'Mackie':
            group_sizes = np.full(groups, reduction)
            group_sizes[-1] += add
        
        #Assigning that grouping to a grouping variable of the same shape as da
        expDims = set(da.dims) - set([red_dim])
        
        ls = np.repeat(np.arange(groups), group_sizes)
        da_groups = xr.DataArray(ls, dims=(red_dim), coords={red_dim: da[red_dim].values})
        values = da.groupby(da_groups).mean()
        boxes = da[red_dim].groupby(da_groups).mean().values
        da_New = xr.DataArray(values, dims=list(da.dims), coords={red_dim: boxes, **{dim: da[dim] for dim in list(expDims)}})   
        
        sumWeights = da_weights.groupby(da_groups).sum()/ reduction
        da_weights = xr.DataArray(sumWeights, dims=list(da.dims), coords={red_dim: boxes, **{dim: da[dim] for dim in list(expDims)}})   
        da_weights.rename('weights')

        da = da_New  
    return da, da_weights

def wetRandSample(das, weights = 'nope', sSize = 100, vals = True):
    import random
    import xarray as xr
    import numpy as np
    '''
    This function take a random sampling of data points from a list of data arrys
    Input data arrays must have the same dimensions
    Weighting is applied by selection - thus low weighted points have a less likely chance of being picked

    Returns a list of lists in the same order that the data arrays were originally assigned
    '''
    
    if isinstance(weights, str):
        weights = xr.ones_like(das[0])

    #Produces a list of all the possible corrdinates
    indices = np.argwhere(np.ones_like(das[0]))
    
    #Produces a flattened list of corresponding weights to those coordinates
    weights_list = weights.to_numpy().ravel()
    
    choice = random.choices(indices, weights= weights_list, k = sSize)  
    weight_value = np.array([weights.isel(dict(zip(das[0].dims, indices))).values.item() for indices in choice])

    retList = []
    weightList = np.array([])
    for da in das:
        element_values = [da.isel(dict(zip(da.dims, indices))).values.item() for indices in choice]
        retList.append(element_values)
    if vals == True:
        return np.array(retList)
    elif vals == False:
        return choice
    elif vals == 'both':
        return np.array(retList), choice

def BinGen(modNameList, incr):
    import pandas as pd
    import numpy as np
    import xarray as xr
    import math
    mmData = pd.DataFrame(columns=['model', 'o500min', 'o500max'])
    for model in modNameList:
        
        tropMin = np.min([np.min(getArray('wap500', 295, model)), np.min(getArray('wap500', 300, model)), np.min(getArray('wap500', 305, model))])
        tropMax = np.max([np.max(getArray('wap500', 295, model)), np.max(getArray('wap500', 300, model)), np.max(getArray('wap500', 305, model))])
        
        mmData = pd.concat([pd.DataFrame({'model': model, 'o500min': tropMin, 'o500max': tropMax}, index=[0]), mmData], ignore_index = True)
    print(mmData)
    bins = np.arange((math.floor(mmData.o500min.min()/incr)*incr), (math.ceil(mmData.o500max.max()/incr)*incr)+incr, incr)
    mnbin = [np.mean([bins[i-1], bins[i]]) for i in range(1,len(bins))] #The midpoints of bins for plotting
    return bins

def newProcess(proName, modNameList, items, location):
    import numpy as np
    import pandas as pd
    import os
    from datetime import datetime

    file_path = "/home/users/chingosa/Functions/Projects/prjList.csv"
    
    if os.path.exists(file_path): #if the prj list exists - read it in
        df = pd.read_csv(file_path)
        if proName in df.proName.to_list(): #Check that its not already in it
            print(f'{proName} is in the prjList already, choose a different name!')
            return
        else:# as long as its not in it
            
            dfNew = pd.DataFrame({'proName': [proName],'date': [datetime.now()], 'status': [0], 'location': location+proName+'.csv'}, index=[0])
            df = pd.concat([df, dfNew], ignore_index=True)
            df.to_csv(file_path, index=False)
            
            df = pd.DataFrame(0, index=modNameList, columns=items)
            df.to_csv(location+proName+'.csv', index=True)
    else:
        print(f"prjList file doesn't exist - making a new one")
        df = pd.DataFrame({'proName': [proName],'date': [datetime.now()], 'status': [0], 'location': location+proName+'.csv'}, index=[0])
        df.to_csv(file_path, index=False)

        df = pd.DataFrame(0, index=modNameList, columns=items)
        df.to_csv(location+proName+'.csv', index=True)

def updateStatus(proName, modName, item, status = 1):
    import numpy as np
    import pandas as pd
    import os
    from datetime import datetime

    file_path = "/home/users/chingosa/Functions/Projects/prjList.csv"
    if not os.path.exists(file_path): #if the prj list exists - read it in
        print('prjList not active - please initiate')
        return
    else:
        df = pd.read_csv(file_path)
        if (proName not in df.proName.tolist()):
            print('project is not in the project list')
            return
        else:
            location = df.location[df.proName == proName].values[0]
            dfSts = pd.read_csv(location, index_col=0)
            if (str(modName) not in dfSts.index.tolist())|(str(item) not in dfSts.columns.tolist()):
                
                print('Either your column or your row name doesnt exist in this project')
                return
            else:
                dfSts.at[str(modName), str(item)] = status
                dfSts.to_csv(location, index = True)

                df.loc[df['proName'] == proName, 'status'] = dfSts.mean().mean()
                df.to_csv(file_path, index = False)

def showStatus(proName):
    import pandas as pd
    import os
    from IPython.display import display

    file_path = "/home/users/chingosa/Functions/Projects/prjList.csv"
    if not os.path.exists(file_path): #if the prj list exists - read it in
        print('prjList not active - please initiate')
        return
    else:
        df = pd.read_csv(file_path)
        if (proName not in df.proName.tolist()):
            print('project is not in the project list')
            return
        else:
            location = df.location[df.proName == proName].values[0]
            dfSts = pd.read_csv(location, index_col=0)
            display(dfSts)

def showPrjList():
    import pandas as pd
    import os
    from IPython.display import display

    file_path = "/home/users/chingosa/Functions/Projects/prjList.csv"
    if not os.path.exists(file_path): #if the prj list exists - read it in
        print('prjList not active - please initiate')
        return
    else:
        df = pd.read_csv(file_path)
        display(df)

def calculate_q3(group):
    return group.quantile(0.75)

def calculate_q1(group):
    return group.quantile(0.25)
    
    

