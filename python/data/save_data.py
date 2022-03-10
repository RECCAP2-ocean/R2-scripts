from pkgutil import get_data


def save_dataset_with_compression(ds, sname):
    """
    Save a dataset with lossy compression. 

    Here we automatically determine the offset and scale factor of the dataset. 
    The min and max values of the dataset are set to the 0.0001 and 99.9999th 
    percentiles respectively. The data is then scaled based on the range and 
    the maximum size of a 16-bit integer. Further, the file is compressed with 
    zlib to a compression level of 1 (fast speed with minimal difference between
    higher values).

    Parameters
    ----------
    ds: xr.Dataset
        The dataset you want to save
    sname: string
        The destination file name of the dataset

    Returns
    -------
    None
    """
    encoding = get_dataset_compression_encoding(ds, maxmin_method='quant', max_percentile=99.9999)
    ds.to_netcdf(sname, encoding=encoding)


def get_dataset_compression_encoding(ds, maxmin_method='max', max_percentile=99.999):
    """
    Creates the encoding to compress data. 
    
    WARNING: 
        this loses precision. If your range is large and your 
        precision is important, then do not use this method. 

    Parameters
    ----------
    ds: xr.Dataset
        The input dataset - does not work with data arrays
    max_method:
    """
    encoding = {}
    for k in ds:
        encoding[k] = get_int16_compression_encoding(
            ds[k],
            max_method=maxmin_method,
            max_percentile=max_percentile)
    return encoding


def get_int16_compression_encoding(da, max_method='max', max_percentile=99.999):
    from numpy import iinfo, int16
    if max_method == 'max':
        x0 = da.min().values
        x1 = da.max().values
    elif max_method.startswith('q') or max_method.startswith('p'):
        x0 = da.quantile(1 - max_percentile / 100).values
        x1 = da.quantile(max_percentile / 100).values
    
    xd = x1 - x0

    i16max = iinfo(int16).max
    offset = x0
    scaler = xd / i16max
    filler = -i16max

    encoding = dict(
        _FillValue=filler,
        add_offset=offset,
        scale_factor=scaler,
        complevel=1,
        zlib=True, 
        dtype='int16',
    )
    return encoding
