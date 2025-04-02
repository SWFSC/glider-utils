import datetime
import logging
import math
import os

import pandas as pd

import esdglider.utils as utils

_log = logging.getLogger(__name__)


def get_path_acoutics(project, deployment, acoustic_path):
    """
    Return a dictionary of acoustic-related paths
    These paths follow the directory structure outlined here:
    https://swfsc.github.io/glider-lab-manual/content/data-management.html

    -----
    Parameters

    project : str
        The project name of the deployment.
        Must be one of: 'FREEBYRD', 'REFOCUS', 'SANDIEGO', 'ECOSWIM'
    deployment : str
        The name of the glider deployment. Eg, amlr01-20210101
    acoustic_path : str
        The path to the top-level folder of the acoustic data.
        This is intended to be the path to the mounted acoustic bucket

    -----
    Returns:
        A dictionary with the relevant acoustic paths

    """

    if not os.path.isdir(acoustic_path):
        raise FileNotFoundError(f"{acoustic_path} does not exist")

    year = utils.year_path(project, deployment)

    acoustic_deployment_path = os.path.join(
        acoustic_path,
        project,
        year,
        deployment,
    )
    if not os.path.isdir(acoustic_deployment_path):
        raise FileNotFoundError(f"{acoustic_deployment_path} does not exist")

    return {
        # "imagedir": os.path.join(acoustic_deployment_path, 'images'),
        "metadir": os.path.join(acoustic_deployment_path, "metadata"),
    }


def echoview_metadata(ds, paths):
    """
    Create metadata files for Echoview acoustics data processing

    Args:
        gdm (GliderDataModel): gdm object created by amlr_gdm
        glider_path (str): path to glider folder
        deployment (str):
        mode (str): deployment-mode string, eg amlr##-YYYYmmdd-delayed

    Returns:
        Tuple of the file names. In order: gps, pitch, roll, depth
    """

    depl = ds.attrs["deployment_name"]
    _log.info(f"Creating echoview metadata for {depl}")

    # Prep - making paths, variables, etc used throughout
    path_echoview = os.path.join(paths["metadir"], "echoview")
    utils.mkdir_pass(paths["metadir"])
    file_echoview_pre = os.path.join(path_echoview, depl)
    utils.mkdir_pass(path_echoview)
    _log.info(f"Writing echoview metadata files to {path_echoview}")

    ds_dt = ds.time.values.astype("datetime64[s]").astype(datetime.datetime)
    mdy_str = [i.strftime("%m/%d/%Y") for i in ds_dt]
    hms_str = [i.strftime("%H:%M:%S") for i in ds_dt]

    # Pitch
    _log.debug("pitch")
    pitch_df = pd.DataFrame(
        {
            "Pitch_date": mdy_str,
            "Pitch_time": hms_str,
            "Pitch_angle": [math.degrees(x) for x in ds["pitch"].values],
        },
    )
    pitch_df.to_csv(f"{file_echoview_pre}-pitch.csv", index=False)

    # Roll
    _log.debug("roll")
    roll_df = pd.DataFrame(
        {
            "Roll_date": mdy_str,
            "Roll_time": hms_str,
            "Roll_angle": [math.degrees(x) for x in ds["roll"].values],
        },
    )
    roll_df.to_csv(f"{file_echoview_pre}-roll.csv", index=False)

    # GPS
    _log.debug("gps")
    gps_df = pd.DataFrame(
        {
            "GPS_date": [i.strftime("%Y-%m-%d") for i in ds_dt],
            "GPS_time": hms_str,
            "Latitude": ds["latitude"].values,
            "Longitude": ds["longitude"].values,
        },
    )
    gps_df.to_csv(f"{file_echoview_pre}-gps.csv", index=False)

    # Depth
    _log.debug("depth")
    depth_df = pd.DataFrame(
        {
            "Depth_date": [i.strftime("%Y%m%d") for i in ds_dt],
            "Depth_time": [f"{i.strftime('%H%M%S')}0000" for i in ds_dt],
            "Depth": ds["depth"].values,
            "repthree": 3,
        },
    )
    depth_file = f"{file_echoview_pre}-depth.evl"
    depth_df.to_csv(depth_file, index=False, header=False, sep="\t")
    utils.line_prepender(depth_file, str(len(depth_df.index)))
    utils.line_prepender(depth_file, "EVBD 3 8.0.73.30735")

    # Wrap up
    _log.info("Finished writing echoview metadata files")
    return file_echoview_pre
