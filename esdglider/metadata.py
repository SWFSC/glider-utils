import os
import logging
import datetime as dt
import glob
import pandas as pd
import numpy as np
import sqlalchemy
import yaml

from importlib.resources import files, as_file
import esdglider.pathutils as pathutils

_log = logging.getLogger(__name__)


# Names of Components in the ESD Glider Database
db_components = {
    "ctd"         : 'CTD', 
    "flbbcd"      : 'flbbcd Fluorometer', 
    "oxygen"      : 'Oxygen Optode', 
    "shadowgraph" : ['Shadowgraph cameras (11cm)', 'Shadowgraph cameras (14cm)'], 
    "glidercam"   : 'Internal Camera Modules', 
    "azfp"        : 'AZFP', 
    "echosounder" : 'Signature 100 compact echosounder'
}
# db_ctd         = 'CTD'
# db_flbbcd      = 'flbbcd Fluorometer'
# db_oxygen      = 'Oxygen Optode'
# db_shadowgraph = ['Shadowgraph cameras (11cm)', 'Shadowgraph cameras (14cm)']
# db_glidercam   = 'Internal Camera Modules'
# db_azfp        = 'AZFP'
# db_echosounder = 'Signature 100 Compact echsounder'

db_factory_cal = ['Factory - Initial', 'Factory - Recal']
# Factory - Iniital
# Factory - Initial
# Factory - Intial
# Factory - Recal
# Factory - recalib


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
    Matches up imagery files with data from pyglider by imagery filename
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
    df['depth']                = ds_sci_interp['depth'].values
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

def fill_instrument(instr_name, prof_vars, devices, x, y):
    """
    instr_name: str
        Name of instrument name, eg 'ctd' or 'oxygen'.
        Name must be a key in db_components
    prof_vars: dict
        Profile variables dictionary
    devices: dict
        Devices dictionary, read in from yaml file
    x: DataFrame
        Pandas dataframe of devices, filtered for Deployment ID
    y: DataFrame
        Pandas dataframe of device calibrations, filtered for Deployment ID
    """

    component_name = db_components[instr_name]
    instr_dict = devices[instr_name]

    instr_dict["serial_number"] = x.loc[x['Component'] == component_name, "Serial_Num"].values[0]

    y_curr = y[y['Component'] == component_name]
    if y_curr.shape[0] > 1:
        raise ValueError(f'Multiple calibrations for {instr_name}')
    elif y_curr.shape[0] == 1:
        instr_dict["calibration_date"] = str(y_curr["Calibration_Date"].values[0])[:10]
        # if instr_name in ["ctd", "flbbcd", "oxygen"]:
        if y_curr["Calibration_Type"].values[0] in db_factory_cal:
            instr_dict["factory_calibrated"] = instr_dict["calibration_date"]
    else:
        _log.info(f"No calibration info for component {instr_name}")

    prof_vars[f"instrument_{instr_name}"] = instr_dict

    return prof_vars


def make_deployment_config(
    deployment: str, project: str, mode: str, out_path: str, 
    db_url=None
):
    """
    deployment : str
        name of the glider deployment. Eg, amlr01-20200101
    project : str
        deployment project name, eg FREEBYRD
    mode : str
        mode for data being generated; either rt or delayed
    out_path : str
        path to which to write the output yaml file
    db_url : str
        The database URL, which is passed to sqlalchemy.create_engine
        to connect to the division database to extract glider info.
        If None (default), no connection attempt will be made

    Returns:
        Full path of the output (written) yaml file
    """

    _log.debug("Reading template yaml files")
    def esdglider_yaml_read(yaml_name):
        with as_file(files('esdglider.data') / yaml_name) as path:
            with open(str(path), 'r') as fin:
                return yaml.safe_load(fin)
    metadata = esdglider_yaml_read('metadata.yml')
    netcdf_vars = esdglider_yaml_read('netcdf-variables-sci.yml')
    prof_vars = esdglider_yaml_read('profile-variables.yml')
    devices = esdglider_yaml_read('glider-devices.yml')


    if db_url is not None:
        _log.debug("connecting to database, with provided URL")
        try:
            engine = sqlalchemy.create_engine(db_url)
            Glider_Deployment = pd.read_sql_table(
                'Glider_Deployment', con = engine, schema = 'dbo')
            vDeployment_Device = pd.read_sql_table(
                'vDeployment_Device', con = engine, schema = 'dbo')
            vDeployment_Device_Calibration = pd.read_sql_table(
                'vDeployment_Device_Calibration', con = engine, schema = 'dbo')
        except:
            raise ValueError('Unable to connect to database and read tablea')

        # Filter for the glider deployment, using the deployment name
        db_depl = Glider_Deployment[Glider_Deployment['Deployment_Name'] == deployment]
        _log.debug("database connection successful")
        # Confirm that exactly one deployment in the db matched deployment name
        if db_depl.shape[0] != 1:
            _log.error(
                'Exactly one row from the Glider_Deployment table ' + 
                f'must match the deployment name {deployment}. ' + 
                f'Currently, {db_depl.shape[0]} rows matched')
            raise ValueError('Invalid Glider_Deployment match')
        
        # Extract the Glider and Glider_Deployment IDs, 
        glider_id = db_depl['Glider_ID'].values[0]
        glider_deployment_id = db_depl['Glider_Deployment_ID'].values[0]
        
        # Get metadata info  
        metadata["deployment_id"] = str(glider_deployment_id)
        metadata["glider_serial"] = " "

        # Filter the Devices table for this deployment
        db_devices = vDeployment_Device[vDeployment_Device['Glider_Deployment_ID'] == glider_deployment_id]
        db_cals = vDeployment_Device_Calibration [vDeployment_Device_Calibration ['Glider_Deployment_ID'] == glider_deployment_id]
        components = db_devices['Component'].values
        
        # Based on the instruments on the glider:
        # 1) Remove netcdf vars from yamls, if necessary
        # 2) Add instrument_ metadata \
        # TODO: turn this section into a loop through the dictionary
        #   Remove items from netcdf_vars based on instrument attribute
        # for key, value in db_components.itmes():
        #     if value in components:
        #         prof_vars[f"instrument_{key}"] = fill_instrument(key, devices, db_devices, db_cals)

        if db_components['ctd'] in components:
            prof_vars = fill_instrument(
                'ctd', prof_vars, devices, db_devices, db_cals)
        else:
            raise ValueError('Glider must have a CTD')
        
        if db_components['flbbcd'] in components:
            prof_vars = fill_instrument(
                'flbbcd', prof_vars, devices, db_devices, db_cals)
        else:
            netcdf_vars.pop('chlorophyll', None)
            netcdf_vars.pop('cdom', None)
            netcdf_vars.pop('backscatter_700', None)

        if db_components['oxygen'] in components:
            prof_vars = fill_instrument(
                'oxygen', prof_vars, devices, db_devices, db_cals)
        else:
            netcdf_vars.pop('oxygen_concentration', None)

        # TODO: how to handle multiple shadowgraph models?
        # if not set(db_components['shadowgraph']).isdisjoint(components):
        #     pass 

        if db_components['glidercam'] in components:
            prof_vars = fill_instrument(
                'glidercam', prof_vars, devices, db_devices, db_cals)
        
        if db_components['azfp'] in components:
            prof_vars = fill_instrument(
                'azfp', prof_vars, devices, db_devices, db_cals)

        if db_components['echosounder'] in components:
            prof_vars = fill_instrument(
                'echosounder', prof_vars, devices, db_devices, db_cals)       


    else:
        _log.info("no database URL provided, and thus no connection attempted")

    deployment_split = pathutils.split_deployment(deployment)

    metadata["deployment_name"] = deployment
    metadata["project"] = project
    metadata["glider_name"] = deployment_split[0]
    if project == "FREEBYRD":
        metadata["sea_name"] = "Southern Ocean"
    elif project in ["ECOSWIM", "SANDIEGO", "REFOCUS"]: 
        metadata["sea_name"] = "Coastal Waters of California"
    else:
        metadata["sea_name"] = "<sea name>"

    deployment_yaml = {
        "metadata" : metadata, 
        "glider_devices" : {}, #has to be an empty dict for pyglider
        "netcdf_variables" : netcdf_vars, 
        "profile_variables" : prof_vars
    }

    yaml_out = os.path.join(out_path, f"{deployment}-{mode}.yml")
    _log.info(f"writing {yaml_out}")
    with open(yaml_out, 'w') as file:
        yaml.dump(deployment_yaml, file)

    return yaml_out
