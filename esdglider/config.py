import logging
import os
from importlib.resources import as_file, files

import pandas as pd
import sqlalchemy
import yaml

import esdglider.utils as utils

_log = logging.getLogger(__name__)


# Names of Components in the ESD Glider Database (table Device_Type)
# NOTE: if changing a key or value, must adjust code below
db_components = {
    "ctd": "CTD",
    "flbbcd": "flbbcd Fluorometer",
    "oxygen": "Oxygen Optode",
    "shadowgraph": "Shadowgraph camera",
    "glidercam": "Glidercam",
    "azfp": "AZFP",
    "echosounder": "Sig 100 compact echosounder",
    "par": "PAR sensor",
    "dmon": "DMON",
    "wispr": "WISPR",
    "hydrophone": "Hydrophone",
}

# Calibration type name from Calibration_Type table
db_factory_cal = ["Factory - Initial", "Factory - Recalibration"]


def instrument_attrs(key, devices, dev_df, cal_df):
    """
    component: str
        Name of database component key, eg 'ctd' or 'oxygen'.
        Name must be a key in db_components
    devices: dict
        Devices dictionary, read in from yaml file
    dev_df: DataFrame
        Pandas dataframe of devices, filtered for Deployment ID
    cal_df: DataFrame
        Pandas dataframe of device calibrations, filtered for Deployment ID
    """

    dev_components = dev_df["Component"]
    component = db_components[key]
    instr = devices[key]

    # Get instrument attributes: serial, make/model, etc
    dev_curr = dev_df[dev_components == component]
    instr["serial_number"] = dev_curr["Serial_Num"].values[0]
    instr["description"] = dev_curr["Device_Description"].fillna("").values[0]
    instr["make_model"] = (
        f"{dev_curr['Manufacturer'].values[0]} {dev_curr['Model'].values[0]}"
    )

    # If CTD, add 'pumped' comment
    if key == "ctd":
        if instr["make_model"] == "Sea-Bird GPCTD":
            instr["comment"] = "Pumped"
        else:
            _log.warning("Unknown CTD make/model")

    # TODO: firmware version

    # Get calibration date, and factory calibration if applicable
    cal_curr = cal_df[cal_df["Component"] == component]
    if cal_curr.shape[0] > 1:
        raise ValueError(f"Multiple calibrations for {component}")
    elif cal_curr.shape[0] == 1:
        instr["calibration_date"] = str(cal_curr["Calibration_Date"].values[0])[:10]
        if cal_curr["Calibration_Type"].values[0] in db_factory_cal:
            instr["factory_calibrated"] = instr["calibration_date"]
    else:
        _log.warning(f"No calibration info for component {component}")

    return instr


def make_deployment_config(deployment_name: str, out_path: str, db_url=None):
    """
    deployment_name : str
        name of the glider deployment. Eg, amlr01-20200101.
        Only need the name of the deployment,
        because the database contains the project
    out_path : str
        path to which to write the output yaml file
    db_url : str
        The database URL, which is passed to sqlalchemy.create_engine
        to connect to the division database to extract glider info.
        If None (default), no connection attempt will be made

    Returns:
        Full path of the output (written) yaml file
    """

    _log.info("Creating config file for deployment %s", deployment_name)
    _log.debug("Reading template yaml files")

    def esdglider_yaml_read(yaml_name):
        with as_file(files("esdglider.data") / yaml_name) as path:
            with open(str(path), "r") as fin:
                return yaml.safe_load(fin)

    metadata = esdglider_yaml_read("metadata.yml")
    netcdf_vars = esdglider_yaml_read("netcdf-variables-sci.yml")
    prof_vars = esdglider_yaml_read("profile-variables.yml")
    devices = esdglider_yaml_read("glider-devices.yml")

    if db_url is not None:
        _log.debug("connecting to database, with provided URL")

        engine = sqlalchemy.create_engine(db_url)
        Glider_Deployment = pd.read_sql_table(
            "vGlider_Deployment",
            con=engine,
            schema="dbo",
        )
        Deployment_Device = pd.read_sql_table(
            "vDeployment_Device",
            con=engine,
            schema="dbo",
        )
        Deployment_Device_Calibration = pd.read_sql_table(
            "vDeployment_Device_Calibration",
            con=engine,
            schema="dbo",
        )

        # Filter for the glider deployment, using the deployment name
        db_depl = Glider_Deployment[
            Glider_Deployment["Deployment_Name"] == deployment_name
        ]
        _log.debug("database connection successful")
        # Confirm that exactly one deployment in the db matched deployment name
        if db_depl.shape[0] != 1:
            _log.error(
                "Exactly one row from the Glider_Deployment table "
                + f"must match the deployment name {deployment_name}. "
                + f"Currently, {db_depl.shape[0]} rows matched",
            )
            raise ValueError("Invalid Glider_Deployment match")

        # Extract various deployment info
        # glider_id = db_depl["Glider_ID"].values[0]
        glider_deployment_id = db_depl["Glider_Deployment_ID"].values[0]
        project = db_depl["Project"].values[0]

        # Get metadata info
        metadata["deployment_id"] = str(glider_deployment_id)

        # Filter the Devices table for this deployment
        db_devices = Deployment_Device[
            Deployment_Device["Glider_Deployment_ID"] == glider_deployment_id
        ]
        db_cals = Deployment_Device_Calibration[
            Deployment_Device_Calibration["Glider_Deployment_ID"]
            == glider_deployment_id
        ]
        components = db_devices["Component"].values

        # Based on the instruments on the glider:
        # 1. Remove netcdf vars from yamls, if necessary
        # 2. Add instrument_ metadata
        instruments = {}
        for key, value in db_components.items():
            if value in components:
                _log.info("Generating config for component %s", value)
                instruments[f"instrument_{key}"] = instrument_attrs(
                    key,
                    devices,
                    db_devices,
                    db_cals,
                )
            else:
                # If we're here, it means this instrument is not on the glider
                _log.debug("No component %s", value)
                if key == "ctd":
                    raise ValueError("Glider must have a CTD")
                if key == "flbbcd":
                    netcdf_vars.pop("chlorophyll", None)
                    netcdf_vars.pop("cdom", None)
                    netcdf_vars.pop("backscatter_700", None)
                if key == "oxygen":
                    netcdf_vars.pop("oxygen_concentration", None)
                    netcdf_vars.pop("oxygen_saturation", None)
                if key == "par":
                    netcdf_vars.pop("par", None)

    else:
        _log.info("no database URL provided, and thus no connection attempted")

    deployment_split = utils.split_deployment(deployment_name)
    metadata["deployment_name"] = deployment_name
    metadata["os_version"] = db_depl["Software_Version"].values[0]
    metadata["project"] = project
    metadata["glider_name"] = deployment_split[0]
    if not any(db_devices["Device_Type"] == "Teledyne Glider Slocum G3"):
        raise ValueError(
            "No device 'Teledyne Glider Slocum G3'. " + "Please add it to the build"
        )
    metadata["glider_serial"] = db_devices.loc[
        db_devices["Device_Type"] == "Teledyne Glider Slocum G3", "Serial_Num"
    ].values[0]

    if project == "FREEBYRD":
        metadata["sea_name"] = "Southern Ocean"
    elif project in ["ECOSWIM", "SANDIEGO", "REFOCUS"]:
        metadata["sea_name"] = "Coastal Waters of California"
    else:
        metadata["sea_name"] = "<sea name>"

    deployment_yaml = {
        "metadata": dict(sorted(metadata.items(), key=lambda v: v[0].upper())),
        "glider_devices": instruments,
        "netcdf_variables": netcdf_vars,
        "profile_variables": prof_vars,
    }

    yaml_out = os.path.join(out_path, f"{deployment_name}.yml")
    _log.info(f"writing {yaml_out}")
    with open(yaml_out, "w") as file:
        yaml.dump(deployment_yaml, file, sort_keys=False)

    return yaml_out
