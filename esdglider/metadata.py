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


def imagery_metadata(ds_eng, ds_sci, imagery_dir, ext = 'jpg'):
    """
    Matches up imagery files with data from gdm object by imagery filename
    Uses interpolated variables (hardcoded in function)
    Returns data frame with metadata information
    
    Args:
        ds_eng (Dataset): xarray Dataset; from engineering timeseries NetCDF
        ds_sci (Dataset): xarray Dataset; from science timeseries NetCDF
        imagery_dir (str): path to folder with images, 
            specifically the 'Dir####' folders
        ext (str, optional): Imagery file extension. Default is 'jpg'.

    Returns:
        DataFrame: pd.DataFrame of imagery metadata
    """
    
    deployment = imagery_dir.split("/")[-2]
    _log.info(f'Creating imagery metadata file for {deployment}')
    _log.info(f"Using images directory {imagery_dir}")

    #--------------------------------------------
    # Checks
    if not os.path.isdir(imagery_dir):
        _log.error(f'imagery_dir ({imagery_dir}) does not exist. ' + 
                   'The imagery metadata file will not be created')
        raise FileNotFoundError(f'Could not find {imagery_dir}')
    else:
        filepaths = glob.glob(f'{imagery_dir}/**/*.{ext}', recursive=True)
        _log.debug(f"Found {len(filepaths)} files with the extension {ext}")
        if len(filepaths) == 0:
            _log.error("Zero image files were found. Did you provide " +
                       "the right path, and use the right file extension?")
            raise ValueError("No files for which to generate metadata")
        imagery_files = [os.path.basename(x) for x in filepaths]
        imagery_files.sort()

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
        imagery_files_dt = [solocam_filename_dt(i, yr_idx) for i in imagery_files]
    except:
        _log.error('Datetimes could not be extracted from imagery filenames, ' + 
                   f'and thus the imagery metadata will not be created')
        raise ValueError('Datetimes could not be extracted from imagery filenames')

    df = pd.DataFrame(data={'img_file': imagery_files, 'img_time': imagery_files_dt})
    # df.to_csv("/home/sam_woodman_noaa_gov/test.csv", index_label="time")

    #--------------------------------------------
    # Create metadata file
    _log.debug("Extracting profile values")
    profile_times = (ds_eng
                 .to_pandas()
                 .reset_index(names='time')
                 .groupby(['profile_index', 'profile_direction'])
                 .agg(min_time=('time', 'min'), max_time=('time', 'max'))
                 .reset_index()
    )
    # Check
    # # TODO: make this into a function
    # # with boolean flag to specify if it should stop on the first instance, or check all
    # overlaps = []
    # for i in range(len(profile_times)):
    #     for j in range(i + 1, len(profile_times)):  # Compare each row with every other row
    #         if (profile_times.loc[i, 'min_time'] <= profile_times.loc[j, 'max_time']) and (profile_times.loc[i, 'max_time'] >= profile_times.loc[j, 'min_time']):
    #             overlaps.append((i, j))  # Store overlapping row indices
    # print(overlaps)

    # Option 1: https://stackoverflow.com/a/44601120
    a = df.img_time.values
    bh = profile_times.max_time.values
    bl = profile_times.min_time.values
    i, j = np.where((a[:, None] >= bl) & (a[:, None] <= bh))
    # d = pd.concat([
    #     df.loc[i, :].reset_index(drop=True),
    #     profile_times.loc[j, :].reset_index(drop=True)
    # ], axis=1)

    # Option 2: Chat
    # a = df.copy()
    # b = profile_times.copy()
    # a['key'] = 0  # Add a dummy key to cross join
    # b['key'] = 0  # Add a dummy key for merging
    # # Perform a cross-join and filter by intervals
    # merged = pd.merge(b, a, on='key').drop('key', axis=1)
    # matches = merged[(merged['img_time'] >= merged['min_time']) & (merged['img_time'] <= merged['max_time'])]

    _log.debug("Interpolating engineering glider data")
    ds_eng_interp = ds_eng.interp(time=df.img_time.values)
    df['latitude']  = ds_eng_interp['latitude'].values
    df['longitude'] = ds_eng_interp['longitude'].values
    df['depth']     = ds_eng_interp['depth'].values
    df['heading']   = ds_eng_interp['heading'].values
    df['pitch']     = ds_eng_interp['pitch'].values
    df['roll']      = ds_eng_interp['roll'].values
    
    _log.debug("Interpolating science glider data")
    ds_sci_interp = ds_sci.interp(time=df.img_time.values)
    df['conductivity']         = ds_sci_interp['conductivity'].values
    df['temperature']          = ds_sci_interp['temperature'].values
    df['pressure']             = ds_sci_interp['pressure'].values
    df['salinity']             = ds_sci_interp['salinity'].values
    df['density']              = ds_sci_interp['density'].values
    df['depth_ctd']            = ds_sci_interp['depth_ctd'].values
    df['oxygen_concentration'] = ds_sci_interp['oxygen_concentration'].values
    df['chlorophyll']          = ds_sci_interp['chlorophyll'].values
    df['cdom']                 = ds_sci_interp['cdom'].values
    df['backscatter_700']      = ds_sci_interp['backscatter_700'].values
    
    #--------------------------------------------
    # Export metadata file
    csv_file = os.path.join(imagery_dir.replace("images", "metadata"), 
                            f'{deployment}-imagery-metadata.csv')
    _log.info(f'Writing imagery metadata to: {csv_file}')
    df.to_csv(csv_file, index=False)

    return df