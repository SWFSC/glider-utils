import collections
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr

_log = logging.getLogger(__name__)


"""
ESD-specific utilities
Mostly helpers for post-processing time series files created using pyglider
"""


# For IOOS-compliant encoding when writing to NetCDF
def to_netcdf_esd(ds: xr.Dataset, outname: str):
    ds.to_netcdf(
        outname,
        "w",
        encoding={
            "time": {
                "units": "seconds since 1970-01-01T00:00:00Z",
                "_FillValue": np.nan,
                "calendar": "gregorian",
                "dtype": "float64",
            },
        },
    )


def findProfiles(stamp: np.ndarray, depth: np.ndarray, **kwargs):
    """
    Function copied exactly (other than pre-commit formatting) from:
    https://github.com/OceanGNS/PGPT/blob/main/scripts/gliderfuncs.py#L196

    Identify individual profiles and compute vertical direction from depth sequence.

    Args:
            stamp (np.ndarray): A 1D array of timestamps.
            depth (np.ndarray): A 1D array of depths.
            **kwargs (optional): Optional arguments including:
                    - length (int): Minimum length of a profile (default=0).
                    - period (float): Minimum duration of a profile (default=0).
                    - inversion (float): Maximum depth inversion between cast segments of a profile (default=0).
                    - interrupt (float): Maximum time separation between cast segments of a profile (default=0).
                    - stall (float): Maximum range of a stalled segment (default=0).
                    - shake (float): Maximum duration of a shake segment (default=0).

    Returns:
            profile_index (np.ndarray): A 1D array of profile indices.
            profile_direction (np.ndarray): A 1D array of vertical directions.
    """
    if not (isinstance(stamp, np.ndarray) and isinstance(depth, np.ndarray)):
        stamp = stamp.to_numpy()
        depth = depth.to_numpy()

    # Flatten input arrays
    depth, stamp = depth.flatten(), stamp.flatten()

    # Check if the stamp is a datetime object and convert to elapsed seconds if necessary
    if np.issubdtype(stamp.dtype, np.datetime64):
        stamp = (stamp - stamp[0]).astype("timedelta64[s]").astype(float)

    # Set default parameter values (did not set type np.timedelta64(0, 'ns') )
    optionsList = {
        "length": 0,
        "period": 0,
        "inversion": 0,
        "interrupt": 0,
        "stall": 0,
        "shake": 0,
    }
    optionsList.update(kwargs)

    validIndex = np.argwhere(
        np.logical_not(np.isnan(depth)) & np.logical_not(np.isnan(stamp)),
    ).flatten()
    validIndex = validIndex.astype(int)

    sdy = np.sign(np.diff(depth[validIndex], n=1, axis=0))
    depthPeak = np.ones(np.size(validIndex), dtype=bool)
    depthPeak[1 : len(depthPeak) - 1,] = np.diff(sdy, n=1, axis=0) != 0
    depthPeakIndex = validIndex[depthPeak]
    sgmtFrst = stamp[depthPeakIndex[0 : len(depthPeakIndex) - 1,]]
    sgmtLast = stamp[depthPeakIndex[1:,]]
    sgmtStrt = depth[depthPeakIndex[0 : len(depthPeakIndex) - 1,]]
    sgmtFnsh = depth[depthPeakIndex[1:,]]
    sgmtSinc = sgmtLast - sgmtFrst
    sgmtVinc = sgmtFnsh - sgmtStrt
    sgmtVdir = np.sign(sgmtVinc)

    castSgmtValid = np.logical_not(
        np.logical_or(
            np.abs(sgmtVinc) <= optionsList["stall"],
            sgmtSinc <= optionsList["shake"],
        ),
    )
    castSgmtIndex = np.argwhere(castSgmtValid).flatten()
    castSgmtLapse = (
        sgmtFrst[castSgmtIndex[1:]]
        - sgmtLast[castSgmtIndex[0 : len(castSgmtIndex) - 1]]
    )
    castSgmtSpace = -np.abs(
        sgmtVdir[castSgmtIndex[0 : len(castSgmtIndex) - 1]]
        * (
            sgmtStrt[castSgmtIndex[1:]]
            - sgmtFnsh[castSgmtIndex[0 : len(castSgmtIndex) - 1]]
        ),
    )
    castSgmtDirch = np.diff(sgmtVdir[castSgmtIndex], n=1, axis=0)
    castSgmtBound = np.logical_not(
        (castSgmtDirch[:,] == 0)
        & (castSgmtLapse[:,] <= optionsList["interrupt"])
        & (castSgmtSpace <= optionsList["inversion"]),
    )
    castSgmtHeadValid = np.ones(np.size(castSgmtIndex), dtype=bool)
    castSgmtTailValid = np.ones(np.size(castSgmtIndex), dtype=bool)
    castSgmtHeadValid[1:,] = castSgmtBound
    castSgmtTailValid[0 : len(castSgmtTailValid) - 1,] = castSgmtBound

    castHeadIndex = depthPeakIndex[castSgmtIndex[castSgmtHeadValid]]
    castTailIndex = depthPeakIndex[castSgmtIndex[castSgmtTailValid] + 1]
    castLength = np.abs(depth[castTailIndex] - depth[castHeadIndex])
    castPeriod = stamp[castTailIndex] - stamp[castHeadIndex]
    castValid = np.logical_not(
        np.logical_or(
            castLength <= optionsList["length"],
            castPeriod <= optionsList["period"],
        ),
    )
    castHead = np.zeros(np.size(depth))
    castTail = np.zeros(np.size(depth))
    castHead[castHeadIndex[castValid] + 1] = 0.5
    castTail[castTailIndex[castValid]] = 0.5

    profileIndex = 0.5 + np.cumsum(castHead + castTail)
    profileDirection = np.empty((len(depth)))
    profileDirection[:] = np.nan

    for i in range(len(validIndex) - 1):
        iStart = validIndex[i]
        iEnd = validIndex[i + 1]
        profileDirection[iStart:iEnd] = sdy[i]

    return profileIndex, profileDirection


def get_fill_profiles(ds, time_vals, depth_vals, **kwargs) -> xr.Dataset:
    """
    Calculate profile index and direction values,
    and fill values and attributes into ds

    ds : `xarray.Dataset`
    time_vals, depth_vals: passed directly to utils.findProfiles

    returns Dataset
    """
    if np.any(np.isnan(ds.depth.values)):
        num_nan = sum(np.isnan(ds.depth.values))
        _log.warning(f"There are {num_nan} nan depth values")

    prof_idx, prof_dir = findProfiles(time_vals, depth_vals, **kwargs)

    idx_comment = (
        "N = inside profile N, N + 0.5 = between profiles N and N + 1. "
        + "Parameters listed as attributes"
    )
    attrs = collections.OrderedDict(
        [
            ("long_name", "profile index"),
            ("units", "1"),
            ("comment", idx_comment),
            ("sources", "time depth"),
            ("method", "esdglider.utils.findProfiles"),
        ]
        + [(key, val) for key, val in kwargs.items()],
    )
    ds["profile_index"] = (("time"), prof_idx, attrs)

    attrs = collections.OrderedDict(
        [
            ("long_name", "glider vertical speed direction"),
            ("units", "1"),
            ("comment", "-1 = ascending, 0 = inflecting or stalled, 1 = descending"),
            ("sources", "time depth"),
            ("method", "esdglider.utils.findProfiles"),
        ],
    )
    ds["profile_direction"] = (("time"), prof_dir, attrs)

    _log.debug(f"There are {np.max(ds.profile_index.values)} profiles")

    return ds


def drop_bogus(ds: xr.Dataset, min_dt: str = "1970-01-01") -> xr.Dataset:
    """
    Remove and/or drop bogus times and values.
    Rows with bogus time or lat/lons are dropped.
    For other bogus values, out of range values are changed to np.nan

    ds: `xarray.Dataset`
        processed glider data
    min_dt: str; default="1970-01-01"
        String represting the minimum datetime to keep.
        Passed to np.datetime64 to be used to filter.
        For instance, '2017-01-01', or '2020-03-06 12:00:00'.

    Returns
    -------
    xarray Dataset
        Dataset with bogus rows rows dropped, and bodus values changed to nan
    """

    # if not (ds_type in ['sci', 'eng']):
    #     raise ValueError('ds_type must be either sci or eng')

    # For out of range or nan time/lat/lon, drop rows
    num_orig = len(ds.time)
    ds = ds.where(ds.time >= np.datetime64(min_dt), drop=True)
    if (num_orig - len(ds.time)) > 0:
        _log.info(
            f"Dropped {num_orig - len(ds.time)} times "
            + f"that were either nan or before {min_dt}",
        )

    num_orig = len(ds.time)
    ll_good = (
        (ds.longitude >= -180)
        & (ds.longitude <= 180)
        & (ds.latitude >= -90)
        & (ds.latitude <= 90)
    )
    ds = ds.where(ll_good, drop=True)
    if (num_orig - len(ds.time)) > 0:
        _log.info(
            f"Dropped {num_orig - len(ds.time)} nan " + "or out of range lat/lons",
        )

    # For science variables, change out of range values to nan
    drop_values = {
        "conductivity": [0, 60],
        "temperature": [-5, 100],
        "pressure": [-2, 1500],
        "chlorophyll": [0, 30],
        "cdom": [0, 30],
        "backscatter_700": [0, 5],
        "oxygen_concentration": [-100, 500],
        "salinity": [0, 50],
        "potential_density": [900, 1050],
        "density": [1000, 1050],
        "potential_temperature": [-5, 100],
    }

    for var, value in drop_values.items():
        if var not in list(ds.keys()):
            _log.debug(f"{var} not present in ds - skipping drop_values check")
            continue
        num_orig = len(ds[var])
        good = (ds[var] >= value[0]) & (ds[var] <= value[1])
        ds[var] = ds[var].where(good, drop=False)
        if num_orig - len(ds[var]) > 0:
            _log.info(
                f"Changed {num_orig - len(ds[var])} {var} values "
                + f"outside range [{value[0]}, {value[1]}] to nan",
            )

    return ds


def get_file_id_esd(ds) -> str:
    """
    ESD's version of pyglider.utils.get_file_id.
    This version does not require a glider_serial
    Make a file id for a Dataset: Id = *glider_name* + "YYYYMMDDTHHMM"
    """

    _log.debug(ds.time)
    if not ds.time.dtype == "datetime64[ns]":
        dt = ds.time.values[0].astype("timedelta64[s]") + np.datetime64("1970-01-01")
    else:
        dt = ds.time.values[0].astype("datetime64[s]")
    _log.debug(f"dt, {dt}")
    id = (
        ds.attrs["glider_name"]
        # + ds.attrs['glider_serial']
        + "-"
        + dt.item().strftime("%Y%m%dT%H%M")
    )
    return id


def data_var_reorder(ds, new_start):
    """
    Reorder the data variables of a dataset

    new_start is a list of the data variable names from ds that
    will be moved to 'first' in the dataset

    Returns ds, with reordered data variables
    """

    ds_vars_orig = list(ds.data_vars)
    if not all([i in ds_vars_orig for i in new_start]):
        _log.error(f"new_start: {new_start}")
        _log.error(f"ds.data_vars: {ds_vars_orig}")
        raise ValueError("All values of new_start must be in ds.data_vars")

    new_order = new_start + [i for i in ds.data_vars if i not in new_start]
    ds = ds[new_order]

    # Double check that all values are present in new ds
    if not (
        all(
            [j in ds_vars_orig for j in new_order]
            + [j in new_order for j in ds_vars_orig],
        )
    ):
        raise ValueError("Error reordering data variables")

    return ds


def datetime_now_utc(format="%Y-%m-%dT%H:%M:%SZ"):
    """
    format : str
        format string; passed to strftime function
        https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

    Returns a string with the current date/time, in UTC,
        controlled by 'format' input
    """
    return datetime.now(timezone.utc).strftime(format)


def encode_times(ds):
    """
    Straight from:
    https://github.com/voto-ocean-knowledge/votoutils/blob/main/votoutils/utilities/utilities.py
    """
    if "units" in ds.time.attrs.keys():
        ds.time.attrs.pop("units")
    if "calendar" in ds.time.attrs.keys():
        ds.time.attrs.pop("calendar")
    ds["time"].encoding["units"] = "seconds since 1970-01-01T00:00:00Z"
    for var_name in list(ds):
        if "time" in var_name.lower() and not var_name == "time":
            for drop_attr in ["units", "calendar", "dtype"]:
                if drop_attr in ds[var_name].attrs.keys():
                    ds[var_name].attrs.pop(drop_attr)
            ds[var_name].encoding["units"] = "seconds since 1970-01-01T00:00:00Z"
    return ds


def split_deployment(deployment):
    """
    Split the deployment string into glider name, and date deployed
    Splits by "-"
    Returns a tuple of the glider name and deployment date
    """
    deployment_split = deployment.split("-")
    deployment_date = deployment_split[1]
    if len(deployment_date) != 8:
        _log.error(
            "The deployment must be the glider name followed by the deployment date",
        )
        raise ValueError(f"Invalid glider deployment date: {deployment_date}")

    return deployment_split


def year_path(project, deployment):
    """
    From the glider project and deployment name (both strings),
    generate and return the year string to use in file paths
    for ESD glider deployments

    For the FREEBYRD project, this will be the year of the second
    half of the Antarctic season. For instance, hypothetical
    FREEBYRD deployments amlr01-20181231 and amlr01-20190101 are both
    during season '2018-19', and thus would return '2019'.

    For all other projects, the value returned is simply the year.
    For example, ringo-20181231 would return 2018,
    and ringo-20190101 would return 2019
    """

    deployment_split = split_deployment(deployment)
    deployment_date = deployment_split[1]
    year = deployment_date[0:4]

    if project == "FREEBYRD":
        month = deployment_date[4:6]
        if int(month) <= 7:
            year = f"{int(year)}"
        else:
            year = f"{int(year) + 1}"

    return year


def mkdir_pass(dir):
    """
    Convenience wrapper to try to make a directory path,
    and pass if it already exists
    """
    _log.debug(f"Trying to make directory {dir}")
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def makedirs_pass(dir):
    """
    Convenience wrapper to try to make a directory path,
    and pass if it already exists
    """
    _log.debug(f"Trying to make directory {dir}")
    if not os.path.exists(dir):
        os.makedirs(dir)


def rmtree(dir, ignore_errors=False):
    """
    Light wrapper around shutil.rmtree
    Checks if directory exists before deleting
    """
    if os.path.isdir(dir):
        _log.info(f"Removing the following directory and all files in it: {dir}")
        shutil.rmtree(dir, ignore_errors=ignore_errors)


def remove_file(file_path):
    """
    Light wrappoer to check if a file exists at file_path,
    and to remove it if so
    """
    if os.path.exists(file_path):
        _log.info(f"Removing file: {file_path}")
        os.remove(file_path)
    else:
        _log.debug(f"No file to remove at: {file_path}")


def find_extensions(dir_path):  # ,  excluded = ['', '.txt', '.lnk']):
    """
    Get all the file extensions in the given directory
    From https://stackoverflow.com/questions/45256250
    """
    extensions = set()
    for _, _, files in Path(dir_path).walk():
        for f in files:
            extensions.add(Path(f).suffix)
            # ext = Path(f).suffix.lower()
            # if not ext in excluded:
            #     extensions.add(ext)
    return extensions


def line_prepender(filename, line):
    """
    Title: prepend-line-to-beginning-of-a-file
    https://stackoverflow.com/questions/5914627
    """

    with open(filename, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip("\r\n") + "\n" + content)


def calc_ts(ds):
    """
    Calculate variables for temperature/salinity plots
    Code adapted from Jacob Partida
    """
    s_lims = (
        np.floor(np.min(ds.salinity) - 0.5),
        np.ceil(np.max(ds.salinity) + 0.5),
    )

    t_lims = (
        np.floor(np.min(ds.potential_temperature) - 0.5),
        np.ceil(np.max(ds.potential_temperature) + 0.5),
    )
    # print(t_lims)
    S = np.arange(s_lims[0], s_lims[1] + 0.1, 0.1)
    T = np.arange(t_lims[0], t_lims[1] + 0.1, 0.1)
    Tg, Sg = np.meshgrid(T, S)
    sigma = gsw.sigma0(Sg, Tg)

    return Sg, Tg, sigma


"""Dictionary for mapping profile_direction values to strings"""
direction_mapping = {1: "Dive", -1: "Climb"}


def calc_regions(ds: xr.Dataset) -> pd.DataFrame:
    """
    Calculate glider profile regions

    From an xarray dataset, likely produced by glider.binary_to_nc,
    create a dataframe with one row for each profile.
    Each row contains columns with summary info about that profile, including:
    - min/max longitude
    - min/max latitude
    - min/max time
    - starting/ending depth

    Processing notes:
    - This function drops removes .5 profile indexes
    - Grouping is only by profile_index. The profile direction is assumed
        to be the mode of the profile_direction column.
        This is because a glider may stall or shake,
        and thus have multiple directions within a profile

    The function logs a warning if there are different numbers of
    dive and climb profiles.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset with glider timeseries data

    Returns
    -------
    pandas Dataframe
        Regions data frame with one row for each dive/climb profile in ds
    """

    # Group by profile index, and summarize other info
    regions_df = (
        ds.to_pandas()
        .reset_index()
        .loc[lambda df: df["profile_index"] % 1 == 0]
        .groupby(["profile_index"], as_index=False)
        .agg(
            profile_direction=(
                "profile_direction", lambda x: x.mode().iloc[0]),
            start_time=("time", "min"),
            end_time=("time", "max"),
            start_depth=("depth", "first"),
            end_depth=("depth", "last"),
            min_lon=("longitude", "min"),
            max_lon=("longitude", "max"),
            min_lat=("latitude", "min"),
            max_lat=("latitude", "max"),
        )
        .assign(
            profile_direction_str=(
                lambda d: d["profile_direction"].map(direction_mapping))
        )
    )

    return regions_df


def check_regions(ds: xr.Dataset, depth_warn: float = 2) -> pd.DataFrame:
    """
    Perform sanity checks on the regions dataframe, including:
    - Check for the same numbers of dive and climb profiles

    Parameters
    ----------
    ds : xarray Dataset
        Dataset with glider timeseries data

    Returns
    -------
    pandas Dataframe
        'between' data frame: one row for each 0.5 profile in ds
    """
    regions_df = calc_regions(ds)

    # Check: the number of dives and climbs are the same
    num_profiles = regions_df.shape[0]
    num_dives = np.count_nonzero(regions_df["profile_direction"] == 1)
    num_climbs = np.count_nonzero(regions_df["profile_direction"] == -1)
    str_divesclimbs = f"dives: {num_dives}; climbs: {num_climbs}"
    _log.debug(f"Total profiles: {num_profiles}; {str_divesclimbs}")

    if num_dives != num_climbs:
        _log.warning(
            "There are different number of dives and climbs: " + f"({str_divesclimbs})",
        )

    # Check: For each group of the 0.5 (i.e., between) profiles, 
    # what is the time and depth split
    # NOTE: a maybe better check would be to get the difference between 
    # subsequent dive/climb profiles
    between_df = (
        ds.to_pandas()
        .reset_index()
        .loc[lambda df: df["profile_index"] % 1 == 0.5]
        .groupby(["profile_index"], as_index=False)
        .agg(
            start_time=("time", "min"),
            end_time=("time", "max"),
            start_depth=("depth", "first"),
            end_depth=("depth", "last"),
        )
        .assign(
            time_diff = lambda d: (d["end_time"] - d["start_time"]), 
            depth_diff = lambda d: abs(d["end_depth"] - d["start_depth"]), 
            prof_loc = lambda d: np.where(d["end_depth"] >= 10, "Deep", "Surface")
        )
    )

    _log.debug(f"The max depth diff is {between_df.depth_diff.max()}")
    _log.debug(f"The max time diff is {between_df.time_diff.max()}")

    if between_df.depth_diff.max() >= depth_warn:
        profs = between_df.profile_index[between_df.depth_diff >= depth_warn]
        _log.warning(
            f"depth difference >= {depth_warn} for the following profile(s): "
            + ", ".join([str(i) for i in profs.values])
        )

        surf_between = between_df[between_df.prof_loc == "Surface"]
        surf_profs = surf_between.profile_index[surf_between.depth_diff >= depth_warn]
        _log.warning(
            f"depth difference >= {depth_warn} for the following surface profile(s): "
            + ", ".join([str(i) for i in surf_profs.values])
        )

    return between_df
