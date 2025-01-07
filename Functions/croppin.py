def croppin(da, dims, red, style = 'Wing'):
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