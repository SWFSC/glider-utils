import datetime
import logging
import math
import os

import pandas as pd
import xarray as xr

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


def regions_evr(
    ds: xr.Dataset,     
    evr_file_pre: str, 
):
    """
    Docs todo
    """
    
    def prof_dir_str(x):
        """Map profile_direction (1/-1 integer) to text descriptor"""
        if x == 1:
            return "Dive"
        elif x == -1:
            return "Climb"
        else:
            raise ValueError("Invalid profile direction integer")

    # Process the dataset top create 'regions dataframe'
    regions_df = (
        ds
        .to_pandas()
        .reset_index()
        # .loc[:, ["time", "latitude", "longitude", "depth", "profile_index", "profile_direction"]]
        .loc[lambda df: df['profile_index'] % 1 == 0]
        .groupby(["profile_index", "profile_direction"], as_index=False)
        .agg(
            min_lon	= ("longitude", "min"), 
            max_lon = ("longitude", "max"), 
            min_lat = ("latitude", "min"), 
            max_lat = ("latitude", "max"), 
            start_time = ("time", "min"), 
            end_time = ("time", "max"), 
            start_date_str = ('time', lambda x: x.min().strftime('%Y%m%d')),
            start_time_str = ('time', lambda x: x.min().strftime('%H%M%S0000')), 
            end_date_str = ('time', lambda x: x.max().strftime('%Y%m%d')),
            end_time_str = ('time', lambda x: x.max().strftime('%H%M%S0000')), 
            profile_direction_str = ('profile_direction', lambda x: prof_dir_str(x.iloc[0])), 
        )
    )

    # Set variables that are consistent throughout the file
    region_dict = {"Dive": 1, "Climb": -1}
    fist_line: str = 'EVRG 7 10.0.298.38422'
    start_depth = -1
    end_depth = 1000

    # For each of the dive and climb regions:
    for r, i in region_dict.items():
        _log.debug(r)
        # Filter for dives/climbs, and set associated variables
        df = regions_df[regions_df["profile_direction"] == i].reset_index(drop=True)
        # df
        # dive_regions_df = regions_df[regions_df["profile_direction"] == 1]

        nrow = len(df)
        region_vec = [fist_line, str(nrow)]

        # Loop through each row and generate the file contents
        for row in df.itertuples():
            idx = row.Index+1
            line1 = (
                f"13 4 {idx} 0 3 -1 1 "
                + f"{row.start_date_str} {row.start_time_str} {start_depth} "
                + f"{row.end_date_str} {row.end_time_str} {end_depth}"
            )
            line5 = (
                f"{row.start_date_str} {row.start_time_str} {end_depth} "
                + f"{row.end_date_str} {row.end_time_str} {end_depth} "
                + f"{row.end_date_str} {row.end_time_str} {start_depth} "
                + f"{row.start_date_str} {row.start_time_str} {start_depth} 1"
            )
            # 'Append' this row's contents to the region vector
            region_vec.extend(['', line1, '0', '0', r, line5, f"Region {idx}"])
            # # line1 = ' '.join(map(str, [13, 4, i+1, 0, 3, -1, 1, date_start, time_start, start_depth, date_end, time_end, end_depth]))
            # # line5 = ' '.join(map(str, [date_start, time_start, end_depth, date_end, time_end, end_depth, date_end, time_end, start_depth, date_start, time_start, start_depth, 1]))
            # # # Region name based on the 'profile' column
            # # reg_name = f'Region {df.loc[i, "profile"]}'
                        # # Append to region_vec
            # region_vec.extend(['', line1, '0', '0', region_class, line5, f"Region {i+1}"])

        with open(f"{evr_file_pre}-{r.lower()}-regions.evr", 'w') as f:
            f.write('\n'.join(region_vec) + '\n')


    # lines_1 = df.apply(
    #     lambda row: f"13 4 {row.name+1} 0 3 -1 1 {row['start_date']} {row['start_time']} {start_depth} {row['end_date']} {row['end_time']} {end_depth}", 
    #     axis=1
    # )
    # lines_5 = df.apply(
    #     lambda row: f"{row['start_date']} {row['start_time']} {end_depth} {row['end_date']} {row['end_time']} {end_depth} {row['end_date']} {row['end_time']} {start_depth} {row['start_date']} {row['start_time']} {start_depth} 1", 
    #     axis=1
    # )

    # region_names = df.apply(lambda x: f"Region {x.name+1}", axis=1)

    # region_vec.extend(lines_1.tolist())
    # region_vec.extend(['0'] * nrow)  # Add '0' for the '0' lines
    # region_vec.extend(['0'] * nrow)  # Add '0' for the '0' lines
    # region_vec.extend(['reg_class'] * nrow)  # Replace with actual class if necessary
    # region_vec.extend(lines_5.tolist())
    # region_vec.extend(region_names.tolist())

    return regions_df


def echoview_metadata(ds: xr.Dataset, paths: dict) -> str:
    """
    Create metadata files for Echoview acoustics data processing

    Parameters
    ----------
    ds : xarray.Dataset
        Science timeseries dataset
    paths : dict
        A dictionary of acoustic file/directory paths
        See get_path_acoutics for the expected key/value pairs

    Returns
    -------
    str
        The filename prefix used for all echoview metadata files
    """

    depl = ds.attrs["deployment_name"]
    _log.info(f"Creating echoview metadata for {depl}")

    # Prep - making paths, variables, etc used throughout
    path_echoview = os.path.join(paths["metadir"], "echoview")
    utils.mkdir_pass(paths["metadir"])
    utils.mkdir_pass(path_echoview)
    file_echoview_pre = os.path.join(path_echoview, depl)
    _log.info(f"Writing echoview metadata files to {path_echoview}")

    ds_dt = ds.time.values.astype("datetime64[s]").astype(datetime.datetime)
    mdy_str = [i.strftime("%m/%d/%Y") for i in ds_dt]
    hms_str = [i.strftime("%H:%M:%S") for i in ds_dt]

    # Regions
    _log.debug(f"regions")
    regions_df = regions_evr(ds, file_echoview_pre)
    regions_df.to_csv(f"{file_echoview_pre}-regions.csv", index=False)

    # Pitch
    _log.debug("pitch")
    pitch_df = pd.DataFrame(
        {
            "Pitch_date": mdy_str,
            "Pitch_time": hms_str,
            "Pitch_angle": [math.degrees(x) for x in ds["pitch"].values],
        },
    )
    pitch_df.to_csv(f"{file_echoview_pre}.pitch.csv", index=False)

    # Roll
    _log.debug("roll")
    roll_df = pd.DataFrame(
        {
            "Roll_date": mdy_str,
            "Roll_time": hms_str,
            "Roll_angle": [math.degrees(x) for x in ds["roll"].values],
        },
    )
    roll_df.to_csv(f"{file_echoview_pre}.roll.csv", index=False)

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
    gps_df.to_csv(f"{file_echoview_pre}.gps.csv", index=False)

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
    depth_file = f"{file_echoview_pre}.depth.evl"
    depth_df.to_csv(depth_file, index=False, header=False, sep="\t")
    utils.line_prepender(depth_file, str(len(depth_df.index)))
    utils.line_prepender(depth_file, "EVBD 3 8.0.73.30735")

    # Wrap up
    _log.info("Finished writing echoview metadata files")
    return file_echoview_pre
