import os
import logging
import pandas as pd
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


def instrument_attrs(instr_name, devices, x, y):
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
    instrument = devices[instr_name]

    instrument["serial_number"] = x.loc[x['Component'] == component_name, "Serial_Num"].values[0]

    y_curr = y[y['Component'] == component_name]
    if y_curr.shape[0] > 1:
        raise ValueError(f'Multiple calibrations for {instr_name}')
    elif y_curr.shape[0] == 1:
        instrument["calibration_date"] = str(y_curr["Calibration_Date"].values[0])[:10]
        # if instr_name in ["ctd", "flbbcd", "oxygen"]:
        if y_curr["Calibration_Type"].values[0] in db_factory_cal:
            instrument["factory_calibrated"] = instrument["calibration_date"]
    else:
        _log.info(f"No calibration info for component {instr_name}")

    return instrument


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
        instruments = {}

        key = 'ctd'
        if db_components[key] in components:
            instruments[f"instrument_{key}"] = instrument_attrs(
                key, devices, db_devices, db_cals)    
        else:
            raise ValueError('Glider must have a CTD')
        
        key = 'flbbcd'
        if db_components[key] in components:
            instruments[f"instrument_{key}"] = instrument_attrs(
                key, devices, db_devices, db_cals)    
        else:
            netcdf_vars.pop('chlorophyll', None)
            netcdf_vars.pop('cdom', None)
            netcdf_vars.pop('backscatter_700', None)

        key = 'oxygen'
        if db_components[key] in components:
            instruments[f"instrument_{key}"] = instrument_attrs(
                key, devices, db_devices, db_cals)    
        else:
            netcdf_vars.pop('oxygen_concentration', None)

        # TODO: how to handle multiple shadowgraph models?
        # if not set(db_components['shadowgraph']).isdisjoint(components):
        #     pass 

        key = 'glidercam'
        if db_components[key] in components:
            instruments[f"instrument_{key}"] = instrument_attrs(
                key, devices, db_devices, db_cals)    
        
        key = 'azfp'
        if db_components[key] in components:
            instruments[f"instrument_{key}"] = instrument_attrs(
                key, devices, db_devices, db_cals)    

        key = 'echosounder'
        if db_components[key] in components:
            instruments[f"instrument_{key}"] = instrument_attrs(
                key, devices, db_devices, db_cals)       


    else:
        _log.info("no database URL provided, and thus no connection attempted")

    deployment_split = pathutils.split_deployment(deployment)
    metadata["deployment_name"] = deployment
    metadata["project"] = project
    metadata["glider_name"] = deployment_split[0]
    metadata["glider_serial"] = ""

    if project == "FREEBYRD":
        metadata["sea_name"] = "Southern Ocean"
    elif project in ["ECOSWIM", "SANDIEGO", "REFOCUS"]: 
        metadata["sea_name"] = "Coastal Waters of California"
    else:
        metadata["sea_name"] = "<sea name>"

    deployment_yaml = {
        "metadata" : dict(sorted(metadata.items(), key = lambda v: v[0].upper())), 
        "glider_devices" : instruments, 
        "netcdf_variables" : netcdf_vars, 
        "profile_variables" : prof_vars
    }

    yaml_out = os.path.join(out_path, f"{deployment}-{mode}.yml")
    _log.info(f"writing {yaml_out}")
    with open(yaml_out, 'w') as file:
        yaml.dump(deployment_yaml, file, sort_keys=False)

    return yaml_out
