import os
import logging
import glob
import numpy as np
import pandas as pd
from datetime import datetime

import esdglider.utils as utils

_log = logging.getLogger(__name__)


def solocam_filename_dt(filename, index_dt, format='%Y%m%d-%H%M%S'):
    """
    Parse solocam (imagery) filename to return associated datetime
    Requires index of start of datetime part of string

    -----
    Parameters

    filename : str
        Full filename
    index_start : int
        The index of the start of the datetime string.
        The datetime runs from this index to this index plus 15 characters
    format : str
        format passed to strptime

    -----
    Returns:
        The datetime extracted from the imagery filename.
        The datetime is returned as a 'datetime64[s]' object
    """

    solocam_substr = filename[index_dt:(index_dt+15)]
    _log.debug(f"datetime substring: {solocam_substr}")
    solocam_dt = datetime.strptime(solocam_substr, format)
    solocam_dt64s = np.datetime64(solocam_dt).astype("datetime64[s]")

    return solocam_dt64s



def get_path_imagery(project, deployment, imagery_path):
    """
    Return a dictionary of imagery-related paths
    These paths follow the directory structure outlined here:
    https://swfsc.github.io/glider-lab-manual/content/data-management.html

    -----
    Parameters

    project : str
        The project name of the deployment.
        Must be one of: 'FREEBYRD', 'REFOCUS', 'SANDIEGO', 'ECOSWIM'
    deployment : str
        The name of the glider deployment. Eg, amlr01-20210101
    mode : str
        Mode of the glider dat being processed.
        Must be either 'rt', for real-time, or 'delayed
    imagery_path : str
        The path to the top-level folder of the imagery data.
        This is intended to be the path to the mounted raw imagery bucket

    -----
    Returns:
        A dictionary with the relevant paths

    """

    # prj_list = ['FREEBYRD', 'REFOCUS', 'SANDIEGO', 'ECOSWIM']
    if not os.path.isdir(imagery_path):
        raise FileNotFoundError(f'{imagery_path} does not exist')

    year = utils.year_path(project, deployment)

    imagery_deployment_path = os.path.join(
        imagery_path, project, year, deployment)
    if not os.path.isdir(imagery_deployment_path):
        raise FileNotFoundError(f'{imagery_deployment_path} does not exist')

    return {
        "imagedir": os.path.join(imagery_deployment_path, 'images'),
        "metadir":  os.path.join(imagery_deployment_path, 'metadata')
    }


def imagery_timeseries(ds, paths, ext = 'jpg'):
    """
    Matches up imagery files with data from pyglider by imagery filename
    Uses interpolated variables (hardcoded in function)
    Returns data frame with imagery+timeseries information

    -----
    Parameters

    ds : xarray Dataset
        from science timeseries NetCDF
    imagery_dir : str
        path to folder with images, specifically the 'Dir####' folders
    ext : str, optional
        Imagery file extension. Default is 'jpg'.

    -----
    Returns:
        DataFrame: pd.DataFrame of imagery timeseries
    """

    deployment = ds.attrs["deployment_name"]
    imagedir = paths['imagedir']
    _log.info(f'Creating imagery metadata file for {deployment}')
    _log.info(f"Using images directory {imagedir}")

    #--------------------------------------------
    # Checks
    if not os.path.isdir(imagedir):
        raise FileNotFoundError(f'{imagedir} does not exist')
    else:
        # NOTE: this should probably be a separate function, and return a tuple
        filepaths = glob.glob(f'{imagedir}/**/*.{ext}', recursive=True)
        _log.debug(f"Found {len(filepaths)} files with the extension {ext}")
        if len(filepaths) == 0:
            _log.error("Zero image files were found. Did you provide " +
                        "the right path, and use the right file extension?")
            raise ValueError("No files for which to generate metadata")
        imagery_files = [os.path.basename(path) for path in filepaths]
        imagery_dirs = [os.path.basename(os.path.dirname(path)) for path in filepaths]

    #--------------------------------------------
    # Extract info from imagery file names
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
        imagery_files_dt = np.array(
            [solocam_filename_dt(i, yr_idx) for i in imagery_files])

    except:
        _log.error('Datetimes could not be extracted from imagery filenames, ' +
                    f'and thus the imagery metadata will not be created')
        raise ValueError('Datetimes could not be extracted from imagery filenames')

    df_data = {'img_file': imagery_files, 'img_dir' : imagery_dirs,
                'time': imagery_files_dt}
    df = pd.DataFrame(data = df_data).sort_values(by='img_file', ignore_index=True)
    # df.to_csv("/home/sam_woodman_noaa_gov/test.csv", index_label="time")

    #--------------------------------------------
    # Create metadata file
    _log.info("Interpolating glider data to image timestamps")
    ds_prof = ds[['profile_index', 'profile_direction']]

    # Must filter df.time for times >= start of ds_prof.time
    img_times = df.time[df.time >= min(ds_prof.time.values)].values
    ds_sel = ds_prof.reindex(time=img_times, method = 'pad')
    df = df.join(ds_sel.to_pandas(), on='time', how='left')

    # For each variable that exists, extract interpolated values
    ds_interp = ds.interp(time=df.time.values)
    vars_list = [
        'latitude', 'longitude', 'depth', 'heading', 'pitch', 'roll',
        'conductivity', 'temperature', 'pressure', 'salinity', 'density',
        'oxygen_concentration', 'chlorophyll', 'cdom'
    ]

    for var in vars_list:
        if not var in list(ds_interp.keys()):
            _log.debug(f"{var} not present in ds - skipping interp")
            continue
        df[var] = ds_interp[var].values

    #--------------------------------------------
    # Export metadata file
    metadir = paths['metadir']
    utils.mkdir_pass(metadir)

    csv_file = os.path.join(metadir, f'{deployment}-imagery-metadata.csv')
    _log.info(f'Writing imagery metadata to: {csv_file}')
    df.to_csv(csv_file, index=False)

    return df
