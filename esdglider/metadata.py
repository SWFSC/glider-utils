import os
import logging
import datetime as dt
import glob
import pandas as pd

_log = logging.getLogger(__name__)


def solocam_filename_dt(filename, index_dt, format='%Y%m%d-%H%M%S'):
    """
    Parse imagery filename to return associated datetime
    Requires index of start of datetime part of string

    filename : str : Full filename
    index_start : int : The index of the start of the datetime string.
        The datetime runs from this index to this index plus 15 characters
    format : str : format passed to strptime

    Returns:
        Datetime object, with the datetime extracted from the imagery filename
    """
    solocam_substr = filename[index_dt:(index_dt+15)]
    _log.debug(f"datetime substring: {solocam_substr}")
    solocam_dt = dt.datetime.strptime(solocam_substr, format)

    return solocam_dt


def imagery_metadata(ds, imagery_path, ext = 'jpg'):
    """
    Matches up imagery files with data from gdm object by imagery filename
    Uses interpolated variables (hardcoded in function)
    Returns data frame with metadata information
    
    Args:
        ds (Dataset): xarray Dataset; from timeseries NetCDF
        imagery_path (str): path to folder with images, 
            specifically the 'Dir####' folders
        ext (str, optional): Imagery file extension. Defaults to 'jpg'.

    Returns:
        DataFrame: DataFrame of imagery metadata
    """
    
    deployment = imagery_path.split("/")[-2]
    _log.info(f'Creating imagery metadata file for {deployment}')
    _log.info(f"Creating from images path {imagery_path}")

    #--------------------------------------------
    # Checks
    if not os.path.isdir(imagery_path):
        _log.error(f'imagery_path ({imagery_path}) does not exist. ' + 
                   'The imagery metadata file will not be created')
        raise FileNotFoundError(f'Could not find {imagery_path}')
    else:
        filepaths = glob.glob(f'{imagery_path}/**/*.{ext}', recursive=True)
        _log.debug(f"Found {len(filepaths)} files with the extension {ext}")
        if len(filepaths) == 0:
            _log.error("Zero image files were found. Did you provide " +
                       "the right path, and use the right file extension?")
            raise ValueError("No files for which to generate metadata")
        imagery_files = [os.path.basename(x) for x in filepaths]
        imagery_files.sort()

    #--------------------------------------------
    # Extract info from imagery file names, and match up with glider data
    _log.debug("Processing imagery file names")

    # Check that all filenames have the same number of characters
    if not len(set([len(i) for i in imagery_files])) == 1:
        _log.warning('The imagery file names are not all the same length, ' + 
            'and thus shuld be checked carefully')

    space_idx = str.index(imagery_files[0], ' ')
    if space_idx == -1:
        _log.error('The imagery file name year index could not be found, ' + 
            'and thus the imagery metadata file cannot be generated')
        raise ValueError("Incompatible file name spaces")
    yr_idx = space_idx + 1   

    try:
        imagery_files_dt = [solocam_filename_dt(i, yr_idx) for i in imagery_files]
    except:
        _log.error('Datetimes could not be extracted from imagery filenames, ' + 
                   f'and thus the imagery metadata will not be created')
        raise ValueError('Datetimes could not be extracted from imagery filenames')

    df = pd.DataFrame(data={'img_file': imagery_files}, index=imagery_files_dt)
    # df.to_csv("/home/sam_woodman_noaa_gov/test.csv", index_label="time")

    _log.debug("Interpolating glider data")
    ds_interp = ds.interp(time=df.index.values)
    df['latitude']  = ds_interp['latitude'].values
    df['longitude'] = ds_interp['longitude'].values
    df['depth']     = ds_interp['depth'].values
    df['heading']   = ds_interp['heading'].values
    df['pitch']     = ds_interp['pitch'].values
    df['roll']      = ds_interp['roll'].values
    
    # TODO: bring in science data. so probs science timeseries?

    #--------------------------------------------
    csv_file = os.path.join(imagery_path.replace("images", "metadata"), 
                            f'{deployment}-imagery-metadata.csv')
    _log.info(f'Writing imagery metadata to: {csv_file}')
    df.to_csv(csv_file, index_label="time")

    return df