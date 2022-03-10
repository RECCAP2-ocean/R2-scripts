"""
A few scripts that help to load RECCAP2-ocean formatted data 

USAGE: 
import load_data as r2

# case 1: loading all data of a model
flist_CESM = r2.get_fnames_recursive_search('../data/Models/3D_ALL/', ['CESM', '_A_'])
ds = r2.open_reccap2_ocean_data(flist_CESM, rename_var_to_model=False, load_data=True)

# case 2: loading a variable for all models (does not work for 3D due to varying depth levels between models)
flist = r2.get_fnames_recursive_search('../data/Surface_CO2/', ['spco2_', '.nc'])
ds = r2.open_reccap2_ocean_data(flist, rename_var_to_model=True, load_data=True)

AUTHOR: 
    LUKE GREGOR (gregorl@ethz.ch)

REQUIRES:
    - numpy
    - pandas
    - xarray
    - dask
    - fuzzywuzzy
"""
import pandas as pd
import xarray as xr
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore', category=FutureWarning)


###############################
## RECCAP SPECIFIC FUNCTIONS ##
###############################
def get_fnames_recursive_search(basedir, patterns=[], exclude=[]):
    """
    Search and match file names in a directory (recursive)
    
    Parameters
    ----------
    basedir: str (must exist as a path)
        the directory that you'd like to search through
    patterns: list of str
        the string patterns that must occur in the files you're looking for
    exclude: list of str
        string patterns you would like to exclude from the filenames
        
    Returns 
    -------
    A list of file names with the full path
    """
    import os
    import re

    flist = []
    for path, subdir, files in os.walk(basedir):
        for fname in files:
            if all([p in fname for p in patterns]):
                has_excluded = [s in fname for s in exclude]
                if not any(has_excluded):
                    flist += os.path.join(path, fname),

    flist = np.sort(flist)
    return flist


def open_reccap2_ocean_data(flist, rename_var_to_model=True, time_lim=slice('1980', '2018'), load_data=True):
    """
    Open RECCAP2-ocean data as a merged netCDF file. 
    Can be used to open a multiple variables from a single model, 
    or a single variable from multiple models
    
    Paramters
    ---------
    flist: list
        A list of files you'd like to import
    rename_var_to_model: bool
        If you're loading a single variable, this must be set to True, 
        if you're loading a single model, this must be set to False
    time_lim: slice
        Defaults to the RECCAP2-ocean model time period
        
    Returns
    -------
    An xr.Dataset with all the variables. 
    To load the data
    
    """
    
    data = []
    for f in flist:
        ds = xr.open_dataset(f, decode_times=False)
        # conform the dataset so that it matches the RECCAP2 protocol
        # some files don't 
        ds_conform = conform_dataset(ds)
        if check_reccap2_format(ds_conform):
            print(f'DROPPED: {f.split("/")[-1]}', end=' ')
            # print the reason why the file was dropped
            check_reccap2_format(ds_conform, verbose=True)
            continue
        else:
            print(f'ADDED: {f.split("/")[-1]}')
            # return only a single variable assigned to the file
            da = get_array_if_only_var(ds_conform)
            # doing this chunking step makes sure that data isnt loaded 
            # when merging making the process a whole lot quicker
            chunks = dict(time=468, depth=100, lat=180, lon=360)
            chunks = {k: chunks[k] for k in chunks if k in da}
            da = da.chunk(chunks)
            # see the docstring
            if rename_var_to_model:
                da = da.rename(ds_conform.model)
            data += da,
    print('Merging files')
    
    ds = xr.merge(data, compat='override').astype('float32')
    if 'time' in ds:
        ds = ds.sel(time=time_lim)
        
    if load_data:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            ds = ds.load()
    return ds


def get_reccap_model_name_from_file_name(ds):
    """
    Uses the file name to guess the RECCAP2 product name
    """
    
    name = (
        ds.encoding['source']
        # get the file name after the last /
        .split('/')[-1]
        # here are some fixes for models that have incorrect naming structure
        # that otherwise breaks the name fetching
        .replace('MPI_SOMFFN', 'MPI-SOMFFN')
        .replace('UOEX_Wat20', 'UOEX-Wat20')
        .replace('LDEO_HPD', 'LDEO-HPD')
        .replace('_REcoM_', '-REcoM-')
        # get the second entry after the underscore (model name)
        .split('_')[1]
        # replace [-.] with _ and
        .replace('-', '_')
        .replace('.', '_'))
    
    return name


def conform_dataset(
    ds,
    default_dim_order=['time', 'depth', 'lat', 'lon'],
):
    """
    Conforms the dataset to the reccap standard (a wrapper for other functions)
    
    Also adds the name of the model
    """
    
    name = get_reccap_model_name_from_file_name(ds)
    
    ds_out = (
        ds
        .squeeze(drop=True)
        .pipe(correct_coord_names)
        .pipe(lon_0E_360E)
        .pipe(coord_05_offset)
        .pipe(correct_depth)
        .pipe(transpose_dims, default=default_dim_order)
        .pipe(decode_times)
        .pipe(get_array_if_only_var)
        .assign_attrs(model=name, fname=ds.encoding['source'])
        .to_dataset()
    )
    ds_out = ds_out.assign_attrs(model=name)
    ds_out.encoding = ds.encoding
    
    return ds_out


###############################
## RECCAP2 FORMATTING CHECKS ##
###############################
def check_reccap2_format(ds, verbose=False):
    """
    Checks if the reccap2 format is met. 
    
    Parameters
    ----------
    ds: xr.Dataset
    verbose: bool
    
    Returns
    -------
    int: 
        1 if failed
        0 if passed
    """
    def vprint(*args, **kwargs):
        if verbose:
            print('(', end='')
            kwargs.update(end=')\n')
            print(*args, **kwargs)
    
    def spatial_coord(da):
        diff = da - (da.values // 1)
        centered = diff == 0.5
        if all(centered):
            return 0
        else:
            return 1
    
    from warnings import warn
    
    if 'source' in ds.encoding:
        name = f"({ds.encoding['source'].split('/')[-1]})"
    else:
        name = ''
        
#     if not has_coords(ds):
#         vprint(f'coordinates missing')
#         return 1
    
    if 'lat' in ds:
        if spatial_coord(ds.lat):
            vprint(f'`lat` not centered on 0.5')
            return 1
            
    if 'lon' in ds:
        if spatial_coord(ds.lon):
            vprint(f'`lon` not centered on 0.5\n{ds.lon.values[:5]}...')
            return 1
        if ds.lon.min() < 0:
            vprint(f'`lon` range is outside 0-360')
            return 1
            
    if 'time' in ds:
        if all([np.issubdtype(t, np.datetime64) for t in ds.time.values]):
            t = ds.time.astype('datetime64[M]')
            delta_days = (ds.time - t).values.astype('timedelta64[D]').astype(float)
            # we expected delta_days to be 14
            if any(delta_days != 14):
                vprint(f"`time` not centered on 15th of the month")
                return 1
            
    return 0
    

def has_coords(ds, checklist=['time', 'lat', 'lon']):
    """
    Check that data has coordinates
    """
    matches = {key: (key in ds.coords) for key in checklist}
    if all(matches.values()):
        return 1
    else:
        return 0
    

###################################
## DECODE RECCAP2 SPECIFIC TIMES ##
###################################
def decode_times(ds):
    """
    Decodes RECCAP2 file times with two methods:
    1) based on year range in the file name 
    2) standard xarray date reading
    
    The first option is prioritised for RECCAP2 files. 
    """
    
    if 'time' not in ds:
        return ds
    
    time_decoders = [
        decode_time_from_fname,
        decode_time_standard,
    ]
    
    time = ds.time.values
    for func in time_decoders:
        try:
            ds = func(ds)
            if not (ds.time.values[0] == time[0]):
                return ds
        except:
            pass
    return ds


def decode_time_from_fname(ds):  
    """
    Uses the file name to get the start and end year from which dates are 
    created. If the created time matches the length of the time variable,
    then the new time is assigned. 
    
    Note that the dataset needs to have be the original import with the 
    encoding properties still present. Alternatively, the the file name
    can be stored as an 'fname' attribute
    
    The file name must contain the start and end year in the format 
    *YYYY-YYYY*
    """
    fname = ds.encoding.get('source', None)
    if fname is None:
        fname = ds.attrs.get('fname', None)
    if fname is None:
        return ds
    
    years = get_years_from_fname(fname)
    
    if years is None:
        return ds
    
    y0, y1 = years
    nyears = int(y1 - y0 + 1)
    nmonth = nyears * 12
    
    if nyears == ds.time.size:
        t = pd.date_range(f'{y0}-01-01', f'{y1}-12-31', freq='1AS')
        t += pd.Timedelta('14D')
        ds = ds.assign_coords(time=t)
        
    if nmonth == ds.time.size:
        t = pd.date_range(f'{y0}-01-01', f'{y1}-12-31', freq='1MS')
        t += pd.Timedelta('14D')
        ds = ds.assign_coords(time=t)
            
    return ds
    

def decode_time_standard(ds):
    """
    Tries to use the standard machinary of xarray to read in the times, but
    ensures that the returned time steps are centered on the 15th of each 
    month. 
    """
    if isinstance(ds.time.values[0], np.datetime64):
        return ds
    else:
        ds = xr.decode_cf(ds)
        dt = np.timedelta64(14, 'D')
        ds = ds.assign_coords(time=ds.time.astype('datetime64[M]') + dt)
        return ds


def get_years_from_fname(fname):
    """
    find the YYYY-YYYY pattern in the file name. Y0 and Y1 are Returned as integers
    """
    import re
    match = re.findall('[12][90][789012][0-9]-[12][90][789012][0-9]', fname)
    if len(match) >= 1:
        t0, t1 = [int(x) for x in match[0].split('-')]
        return t0, t1
    else:
        return None


#############################
## FIX SPATIAL COORDINATES ##
#############################
def lon_180W_180E(ds, lon_name='lon'):
    """
    Regrid the data from [-180 : 180] from [0 : 360]
    """
    
    lon180 = (ds[lon_name] - 180) % 360 - 180
    
    return (
        ds
        .assign_coords(**{lon_name: lon180})
        .sortby(lon_name))


def lon_0E_360E(ds, lon_name='lon'):
    """
    Regrid the data from [0 : 360] from [-180 : 180] 
    """
    
    if lon_name not in ds:
        return ds
    
    lon = ds[lon_name].values
    lon360 = lon % 360
    # save some work by checking if already 0-360
    if (lon360 != lon).any():
        ds = (
            ds
            .assign_coords(**{lon_name: lon360})
            .sortby(lon_name))
        return ds
    else:
        return ds

    
def coord_05_offset(ds, center=0.5, coord_name='lon'):
    """
    Will interpolate data if the grid centers are offset. 
    Only works for 1deg data
    
    Parameters
    ----------
    ds: xr.Dataset
        the dataset with a coordinate variable variable
    center: float
        the desired center point of the grid points between 0 - 1
    coord_name: str [lon]
        the name of the coordinate 
        
    Returns
    -------
    xr.Dataset: interpolated onto the new grid with the new
        coord being the old coord + center
    """
    
    center = center - (center // 1)
    if has_coords(ds):
        coord = ds[coord_name].values
        mod = coord - (coord // 1)
        # use the modulus to determine if grid centers are correct
        if any(mod != center):
            ds = ds.interp({coord_name: coord + center})
            
    return ds
    

def transpose_dims(ds, default=['time', 'depth', 'lat', 'lon'], other_dims_before=True):
    """
    Ensures that dimensions are always in the given order. 
    Can specify if remaining dimensions should be ordered before 
    or after the default dimensions.
    """
    dims = set(list(ds.dims))
    default = [d for d in default if d in dims]
    default_set = set(default)
    other = dims - default_set
    
    if other_dims_before:
        new_order = list(other) + list(default)
    else:
        new_order = list(default) + list(other)
    
    ds = ds.transpose(*new_order)
    
    return ds


def correct_depth(ds):
    """converts cemtimeters to meters"""
    if 'depth' in ds:
        if ds.depth.attrs.get('units', '') == 'centimeters':
            ds = ds.assign_coords(depth=ds.depth / 100)
           
    return ds


###############################
## ESTIMATE COORDINATE NAMES ##
###############################
def correct_coord_names(
    ds, 
    match_dict=dict(
        time=["month", "time", "t"],
        depth=["depth", "z", "lev", "z_t", "z_l"],
        lat=["lat", "latitude", "y"], 
        lon=["lon", "longitude", "x"])
):
    """
    Uses a fuzzy function to rename coordinate names so that they match 
    the standard coordinate names names
    
    Parameters
    ----------
    ds: xr.Dataset
    match_dict: dict
        A dictionary where the keys are the desired coordinate/dimension names
        The values are the nearest guesses for possible names. Note these do 
        not have to match the possible names perfectly. 
        
    Returns
    -------
    xr.Dataset: with renamed coordinates that match the keys from match_dict
    """
    coord_keys = list(set(list(ds.coords) + list(ds.dims)))
    coord_renames = guess_coords_from_column_names(coord_keys, match_dict=match_dict)
    
    if any(coord_renames):
        str_renames = str(coord_renames).replace("'", "").replace(':', ' -->')
        ds = ds.rename(coord_renames)
    
    return ds


def guess_coords_from_column_names(
    column_names, 
    match_dict=dict(
        time=["month", "time", "t", "date"],
        depth=["depth", "z"],
        lat=["lat", "latitude", "y"], 
        lon=["lon", "longitude", "x"],)
):
    """
    Takes a list of column names and guesses 
    """
    
    coord_names = {}
    for col in column_names:
        est_name = estimate_name(col, match_dict)
        if est_name != col:
            coord_names[col] = est_name

    coord_names = drop_worst_duplicates_from_rename_dict(coord_names)
    return coord_names


def drop_worst_duplicates_from_rename_dict(rename_dict):
    """
    Will remove the weakest matching key-value pair where the value is duplicate. 
    
    Parameters
    ----------
    rename_dict: dict 
        keys are the original values and keys are the new names. 
        
    Returns
    -------
    dict: the same dictionary with the weakest matching duplicates removed
        
    """
    
    names = pd.Series(rename_dict)
    duplicates = names.duplicated()
    duplicated_coords = names[duplicates].unique()
    
    drop_duplicates = []
    for duplicate_name in duplicated_coords:
        original_columns = names[names == duplicate_name].index.tolist()
        
        ratios = fuzzy_matching(duplicate_name, original_columns).mean(axis=1)
        best_match = ratios.idxmax()
        
        original_columns.remove(best_match)
        drop_duplicates += original_columns

    names_wo_duplicates = names.drop(drop_duplicates)
    
    return dict(names_wo_duplicates)


def estimate_name(name, match_dict):
    """
    Gets the closest match for the name from the match dictionary.
    
    Uses FuzzyWuzzy library to find the nearest match for values in 
    the match_dict. They key of the nearest match will be assigned as
    the new name. 
    
    Parameters
    ----------
    name: str
        A string name that you'd like to find a match with 
    match_dict: dict
        Keys are the new name. Values can be list or string and are
        the possible near matches. 
    threshold: int [75]
        A new name will not be assigned if the ratio does not exceed this
        value
        
    Returns
    -------
    str: 
        either the original name if no strong matches, or a key from the 
        match_dict for the best matching value pair. 
    
    """

    best_ratio = 0
    best_name = ""
    for key in match_dict:
        ratios = fuzzy_matching(name, match_dict[key]).mean(axis=1)
        if ratios.max() > best_ratio:
            best_name = key
            best_ratio = ratios.max()
            
    if best_ratio > 75:
        return best_name
    else:
        return name
    

def fuzzy_matching(s, possible_matches):
    """
    Does fuzzy matching of a string with a list of possible strings
    
    Paramters
    ---------
    s: str
        The string you'd like to find the closest match with
    possible_matches: list
        A list of strings that could be a match for s
        
    Returns
    -------
    pd.dataframe
        fuzzy match ratios with partial_ratios and ratios where
        0 is the min and 100 is the max. The columns are the two
        types of matching algos and the rows are the entries from
        the possible matches. 
        
    Note
    ----
    This is a wrapper around fuzz_ratios that does case insensitive 
    matching. 
    """
    from fuzzywuzzy import fuzz
    ratios = {}
    for func in [fuzz.partial_ratio, fuzz.ratio]:
        name = func.__name__
        ratios[name] = fuzz_ratios(s, possible_matches, func)
    return pd.DataFrame(ratios)   
    
    
def fuzz_ratios(s, possible_matches, func=None):
    """
    Does fuzzy matching of a string with a list of strings
    
    Parameters
    ----------
    s: str
        the string you'd like to match with
    possible_matches: list
        a list of strings that could match with s
    func: [None|callable]
        if None, then will default to fuzzywuzzy.fuzz.partial_ratio
        accepts other fuzzywuzzy functions that return ratioss

    Returns
    -------
    dict: 
        ratios for each of the possible_matches entries
    """
    if func is None:
        from fuzzywuzzy.fuzz import partial_ratio as func
    
    x = s.lower()
    if isinstance(possible_matches, list):
        y = possible_matches
    if isinstance(possible_matches, str):
        y = [possible_matches]
    
    ratios = {m: func(m.lower(), x) for m in y}
    return ratios


######################
## HELPER FUNCTIONS ## 
######################
def get_array_if_only_var(ds):
    """
    If a variable name is in the file name, then only that variable will be 
    returned. An xr.DataArray is returned (not a Dataset)
    """
    fpath = ds.encoding['source']
    fname = fpath.split('/')[-1]
    
    data_vars = list(ds.data_vars)
    n_vars = len(data_vars)
    # if more than one data variable, return the variable that 
    # occurs in the file name
    if n_vars != 1:  
        abridged_fname = "_".join(fname.split("_")[:2])
        for var in data_vars:
            if var in abridged_fname:
                ds = ds[var]
    if n_vars == 1:
        ds = ds[data_vars[0]]
        
    ds.encoding['source'] = fname
            
    return ds


def return_original_if_failed(func):
    """
    A helper function that will return the original input
    if the function failed. Useful for xr.Dataset.pipe(function)
    """
    
    from functools import wraps
    @wraps(func)
    def wrapper(ds, **kwargs):
        try:
            ds = func(ds, **kwargs)
        except:
            funcname = func.__name__
            print(f'`{funcname}` failed, returning original input')
        return ds
    return wrapper
