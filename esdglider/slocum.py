import os
import logging
import numpy as np
import xarray as xr
import yaml
import netCDF4
import importlib

import pyglider.slocum as pgslocum
import pyglider.ncprocess as pgncprocess
import pyglider.utils as pgutils

import esdglider.utils as utils


_log = logging.getLogger(__name__)


def get_path_engyaml():
    """
    Get and return the path to the yaml with engineering NetCDF variables
    Returns the path, so as to be able to pass to binary_to_timeseries
    """

    ref = importlib.resources.files("esdglider.data") / "deployment-eng-vars.yml"
    with importlib.resources.as_file(ref) as path:
        return str(path)


def get_path_deployment(project, deployment, mode, deployments_path, config_path):
    """
    Return a dictionary of paths for use by other esdglider functions.
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

    deployments_path : str
        The path to the top-level folder of the glider data.
        This is inteded to be the path to the mounted glider deployments bucket

    config_path : str
        The path to the directory that contains the yaml with the
        deployment config

    -----
    Returns:
        A dictionary with the relevant paths
    """

    prj_list = ["FREEBYRD", "REFOCUS", "SANDIEGO", "ECOSWIM"]
    if not os.path.isdir(deployments_path):
        _log.error(f"deployments_path ({deployments_path}) does not exist")
        return
    else:
        dir_expected = prj_list + ["cache"]
        if not all(x in os.listdir(deployments_path) for x in dir_expected):
            _log.warning(
                f"The expected folders ({', '.join(dir_expected)}) "
                + f"were not found in the provided directory ({deployments_path}). "
                + "Did you provide the right path via deployments_path?"
            )

    year = utils.year_path(project, deployment)

    glider_path = os.path.join(deployments_path, project, year, deployment)
    if not os.path.isdir(glider_path):
        _log.error(f"glider_path ({glider_path}) does not exist")
        return

    # if write_imagery:
    #     if not os.path.isdir(imagery_path):
    #         _log.error('write_imagery is true, and thus imagery_path ' +
    #                       f'({imagery_path}) must be a valid path')
    #         return

    cacdir = os.path.join(deployments_path, "cache")
    binarydir = os.path.join(glider_path, "data", "binary", mode)
    deploymentyaml = os.path.join(config_path, f"{deployment}.yml")
    # deploymentyaml = os.path.join(glider_path, 'config',
    #     f"{deployment_mode}.yml")
    engyaml = get_path_engyaml()

    ncdir = os.path.join(glider_path, "data", "nc")

    tsdir = os.path.join(ncdir, "timeseries")
    profdir = os.path.join(ncdir, "ngdac", mode)
    griddir = os.path.join(ncdir, "gridded")
    plotdir = os.path.join(glider_path, "plots")

    return {
        "cacdir": cacdir,
        "binarydir": binarydir,
        "deploymentyaml": deploymentyaml,
        "engyaml": engyaml,
        "tsdir": tsdir,
        "profdir": profdir,
        "griddir": griddir,
        "plotdir": plotdir,
    }


def binary_to_nc(
    deployment, mode, paths, min_dt, write_timeseries=True, write_gridded=True
):
    """
    Process binary ESD glider data to timeseries and/or gridded netCDF files

    The contents of this function used to just be in scripts/binary_to_nc.py.
    They were moved to this structure for easier development and debugging

    -----
    Parameters

        deployment : str
        The name of the glider deployment. Eg, amlr01-20210101

    mode : str
        Mode of the glider dat being processed.
        Must be either 'rt', for real-time, or 'delayed

    paths : dict
        A dictionary of file/directory paths for various processing steps.
        Intended to be the output of esdglider.slocum.paths_esd_gcp()
        See this function for the expected key/value pairs

    min_dt : datetime64, or object that can be converted to datetime64
        See utils.drop_bogus; default is '2017-01-01'.
        All timestamps from before this value will be dropped

    write_timeseries, write_gridded : bool
        Should the timeseries and gridded, respectively,
        xarray DataSets be both created and written to files?
        Note: if True then already-existing files will be clobbered

    -----
    Returns

    A tuple of the filenames of the various netCDF files, as strings.
    In order: the engineering and science timeseries,
    and the 1m and 5m gridded files

    """

    # Choices (delayed, rt) specified in arg input
    if mode == "delayed":
        binary_search = "*.[D|E|d|e]bd"
    else:
        binary_search = "*.[S|T|s|t]bd"

    # --------------------------------------------
    # Check file and directory paths
    tsdir = paths["tsdir"]
    deploymentyaml = paths["deploymentyaml"]

    # Get deployment and thus file name from yaml file
    with open(deploymentyaml) as fin:
        deployment_ = yaml.safe_load(fin)
        deployment_name = deployment_["metadata"]["deployment_name"]
    if deployment_name != deployment:
        raise ValueError(
            f"Provided deployment ({deployment}) is not the same as "
            + f"the deploymentyaml deployment_name ({deployment_name})"
        )

    # --------------------------------------------
    # TODO: handle compressed files, if necessary.
    # Although maybe this should be in another function?

    # --------------------------------------------
    # Timeseries
    if write_timeseries:
        if not os.path.exists(tsdir):
            _log.info(f"Creating directory at: {tsdir}")
            os.makedirs(tsdir)

        if not os.path.isfile(deploymentyaml):
            raise FileNotFoundError(f"Could not find {deploymentyaml}")

        # Engineering - uses m_depth as time base
        _log.info("Generating engineering timeseries")
        outname_tseng = pgslocum.binary_to_timeseries(
            paths["binarydir"],
            paths["cacdir"],
            tsdir,
            [deploymentyaml, paths["engyaml"]],
            search=binary_search,
            fnamesuffix=f"-{mode}-eng",
            time_base="m_depth",
            profile_filt_time=None,
        )

        _log.info("Post-processing engineering timeseries")
        tseng = xr.load_dataset(outname_tseng)
        tseng = postproc_eng_timeseries(tseng, mode, min_dt=min_dt)
        tseng.to_netcdf(outname_tseng, encoding=utils.encoding_dict)
        _log.info(f"Finished eng timeseries postproc: {outname_tseng}")

        # Science - uses sci_water_temp as time_base sensor
        _log.info("Generating science timeseries")
        outname_tssci = pgslocum.binary_to_timeseries(
            paths["binarydir"],
            paths["cacdir"],
            tsdir,
            deploymentyaml,
            search=binary_search,
            fnamesuffix=f"-{mode}-sci",
            time_base="sci_water_temp",
            profile_filt_time=None,
        )

        _log.info("Post-processing science timeseries")
        tssci = xr.load_dataset(outname_tssci)
        tssci = postproc_sci_timeseries(tssci, mode, min_dt=min_dt)
        tssci.to_netcdf(outname_tssci, encoding=utils.encoding_dict)
        _log.info(f"Finished sci timeseries postproc: {outname_tssci}")

        num_profiles_eng = len(np.unique(tseng.profile_index.values))
        num_profiles_sci = len(np.unique(tssci.profile_index.values))
        if num_profiles_eng != num_profiles_sci:
            _log.warning(
                "The eng and sci timeseries have different total numbers of profiles"
            )
            _log.debug(f"Number of eng profiles: {num_profiles_eng}")
            _log.debug(f"Number of sci profiles: {num_profiles_sci}")

    else:
        _log.info("Not writing timeseries")
        outname_tseng = os.path.join(tsdir, f"{deployment}-{mode}-eng.nc")
        outname_tssci = os.path.join(tsdir, f"{deployment}-{mode}-sci.nc")

    # --------------------------------------------
    # Gridded data, 1m and 5m
    # TODO: filter to match SOCIB?
    if write_gridded:
        if not os.path.isfile(outname_tssci):
            raise FileNotFoundError(f"Could not find {outname_tssci}")

        _log.info("Generating 1m gridded data")
        outname_1m = pgncprocess.make_gridfiles(
            outname_tssci,
            paths["griddir"],
            deploymentyaml,
            dz=1,
            fnamesuffix=f"-{mode}-1m",
        )

        _log.info("Generating 5m gridded data")
        outname_5m = pgncprocess.make_gridfiles(
            outname_tssci,
            paths["griddir"],
            deploymentyaml,
            dz=5,
            fnamesuffix=f"-{mode}-5m",
        )

    else:
        _log.info("Not writing gridded data")
        outname_1m = os.path.join(paths["griddir"], f"{deployment}_grid-{mode}-1m.nc")
        outname_5m = os.path.join(paths["griddir"], f"{deployment}_grid-{mode}-5m.nc")

    # --------------------------------------------
    # Write imagery metadata file
    # if write_imagery:
    #     _log.info("write_imagery is True, and thus writing imagery metadata file")
    #     if mode == 'rt':
    #         _log.warning('You are creating imagery file metadata ' +
    #             'using real-time data. ' +
    #             'This may result in inaccurate imagery file metadata')
    #     amlr_imagery_metadata(
    #         gdm, deployment, glider_path,
    #         os.path.join(imagery_path, 'gliders', args.ugh_imagery_year, deployment)
    #     )

    # --------------------------------------------
    return outname_tseng, outname_tssci, outname_1m, outname_5m


def postproc_attrs(ds, mode):
    """
    Update attrbites of xarray DataSet ds
    Used for both engineering and science timeseries

    Returns the ds Dataset with updated attributes
    """

    try:
        del ds.attrs["glider_serial"]
    except KeyError:
        _log.warning("Unable to delete glider_serial attribute")
        pass

    # Update attrs set by pyglider functions, now that drop_bogus has been run
    good = ~np.isnan(ds.latitude.values + ds.longitude.values)
    if np.any(good):
        ds.attrs["geospatial_lat_max"] = np.max(ds.latitude.values[good])
        ds.attrs["geospatial_lat_min"] = np.min(ds.latitude.values[good])
        ds.attrs["geospatial_lon_max"] = np.max(ds.longitude.values[good])
        ds.attrs["geospatial_lon_min"] = np.min(ds.longitude.values[good])
    else:
        ds.attrs["geospatial_lat_max"] = np.nan
        ds.attrs["geospatial_lat_min"] = np.nan
        ds.attrs["geospatial_lon_max"] = np.nan
        ds.attrs["geospatial_lon_min"] = np.nan

    ds = pgutils.get_distance_over_ground(ds)

    ds.attrs["id"] = utils.get_file_id_esd(ds)
    ds.attrs["title"] = ds.attrs["id"]
    ds.attrs["deployment_start"] = str(ds.time.values[0])[:19]
    ds.attrs["deployment_end"] = str(ds.time.values[-1])[:19]
    dt = ds.time.values
    ds.attrs["time_coverage_start"] = "%s" % dt[0]
    ds.attrs["time_coverage_end"] = "%s" % dt[-1]

    # ESD updates, or fixes of pyglider attributes
    ds.attrs["standard_name_vocabulary"] = "CF Standard Name Table v72"
    ds.attrs["history"] = (
        f"{np.datetime64('now')}Z: netCDF files created using: "
        + f"pyglider v{importlib.metadata.version("pyglider")}; "
        + f"esdglider v{importlib.metadata.version("esdglider")}"
    )
    ds.attrs["processing_level"] = (
        "Minimal data screening. "
        + "Data provided as is, with no expressed or implied assurance "
        + "of quality assurance or quality control."
    )

    if mode == "delayed":
        ds.attrs["title"] = ds.attrs["title"] + "-delayed"

    return ds


def postproc_eng_timeseries(ds, mode, min_dt):
    """
    Post-process engineering timeseries, including:
        - Removing CTD vars
        - Calculating profiles using depth (m_depth)
        - Updating attributes

    ds : `xarray.Dataset`
        engineering Dataset, usually passed from binary_to_nc.py
    min_dt: passed to drop_bogus_times

    returns post-processed Dataset
    """

    _log.debug(f"begin eng postproc: ds has {len(ds.time)} values")

    # Drop CTD variables required or created by binary_to_timeseries
    ds = ds.drop_vars(
        [
            "depth",
            "conductivity",
            "temperature",
            "pressure",
            "salinity",
            "potential_density",
            "density",
            "potential_temperature",
        ]
    )

    # With depth (CTD) gone, rename depth_measured
    ds = ds.rename({"depth_measured": "depth"})

    # Remove times < min_dt
    ds = utils.drop_bogus(ds, min_dt)

    # Calculate profiles using measured depth
    if np.any(np.isnan(ds.depth.values)):
        num_nan = sum(np.isnan(ds.depth.values))
        _log.warning(f"There are {num_nan} nan depth values")
    ds = utils.get_fill_profiles(ds, ds.time.values, ds.depth.values)

    # Reorder data variables
    new_start = ["latitude", "longitude", "depth", "profile_index"]
    ds = utils.data_var_reorder(ds, new_start)

    # Update comment
    if "comment" not in ds.attrs:
        ds.attrs["comment"] = "engineering-only time series"
    elif not ds.attrs["comment"].strip():
        ds.attrs["comment"] = "engineering-only time series"
    else:
        ds.attrs["comment"] = ds.attrs["comment"] + "; engineering-only time series"

    _log.debug(f"end eng postproc: ds has {len(ds.time)} values")
    ds = postproc_attrs(ds, mode)

    return ds


def postproc_sci_timeseries(ds, mode, min_dt):
    """
    Post-process science timeseries, including:
        - remove bogus times. Eg, 1970, or before deployment start date
        - Calculating profiles using depth (derived from ctd's pressure)

    ds : `xarray.Dataset`
        science Dataset, usually passed from binary_to_nc.py
    min_dt: passed to drop_bogus_times

    returns post-processed Dataset
    """

    _log.debug(f"begin sci postproc: ds has {len(ds.time)} values")

    # Remove times < min_dt
    ds = utils.drop_bogus(ds, min_dt)

    # TODO: redo pyglider metadata things that are changed by min_dt

    # Calculate profiles, using the CTD-derived depth values
    # TODO: update this to play nice with eng timeseries for rt data?
    ds = utils.get_fill_profiles(ds, ds.time.values, ds.depth.values)

    # Reorder data variables
    new_start = [
        "latitude",
        "longitude",
        "depth",
        "profile_index",
        "conductivity",
        "temperature",
        "pressure",
        "salinity",
        "density",
        "potential_temperature",
        "potential_density",
    ]
    ds = utils.data_var_reorder(ds, new_start)

    _log.debug(f"end sci postproc: ds has {len(ds.time)} values")
    ds = postproc_attrs(ds, mode)

    return ds


def ngdac_profiles(inname, outdir, deploymentyaml, force=False):
    """
    ESD's version of extract_timeseries_profiles, from:
    https://github.com/c-proof/pyglider/blob/main/pyglider/ncprocess.py#L19

    Extract and save each profile from a timeseries netCDF.

    ------
    Parameters

    inname : str or Path
        netcdf file to break into profiles
    outdir : str or Path
        directory to place profiles
    deploymentyaml : str or Path
        location of deployment yaml file for the netCDF file.  This should
        be the same yaml file that was used to make the timeseries file.
    force : bool, default False
        Force an overwite even if profile netcdf already exists

    ------
    Returns
        Nothing
    """
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    with open(deploymentyaml) as fin:
        deployment = yaml.safe_load(fin)

    # ESD: include all instrument vars
    # deployment["glider_devices"]
    instrument_meta = deployment["glider_devices"]
    instrument_str = ",".join(list(instrument_meta.keys()))

    meta = deployment["metadata"]
    with xr.open_dataset(inname) as ds:
        _log.info("Extracting profiles: opening %s", inname)
        trajectory = utils.get_file_id_esd(ds).encode()
        trajlen = len(trajectory)

        # TODO: do floor like oceanGNS??
        profiles = np.unique(ds.profile_index)
        profiles = [p for p in profiles if (~np.isnan(p) and not (p % 1) and (p > 0))]
        for p in profiles:
            ind = np.where(ds.profile_index == p)[0]
            dss = ds.isel(time=ind)
            outname = outdir + "/" + utils.get_file_id_esd(dss) + ".nc"
            _log.info("Checking %s", outname)
            if force or (not os.path.exists(outname)):
                # this is the id for the whole file, not just this profile..
                dss["trajectory"] = trajectory
                # trajlen = len(pgutils.get_file_id(ds).encode())
                dss["trajectory"].attrs["cf_role"] = "trajectory_id"
                dss["trajectory"].attrs["comment"] = (
                    "A trajectory is a single"
                    "deployment of a glider and may span multiple data files."
                )
                dss["trajectory"].attrs["long_name"] = "Trajectory/Deployment Name"

                # profile-averaged variables....
                profile_meta = deployment["profile_variables"]
                if "water_velocity_eastward" in dss.keys():
                    dss["u"] = dss.water_velocity_eastward.mean()
                    dss["u"].attrs = profile_meta["u"]

                    dss["v"] = dss.water_velocity_northward.mean()
                    dss["v"].attrs = profile_meta["v"]
                elif "u" in profile_meta:
                    dss["u"] = profile_meta["u"].get("_FillValue", np.nan)
                    dss["u"].attrs = profile_meta["u"]

                    dss["v"] = profile_meta["v"].get("_FillValue", np.nan)
                    dss["v"].attrs = profile_meta["v"]
                else:
                    dss["u"] = np.nan
                    dss["v"] = np.nan

                dss["profile_id"] = np.int32(p)
                dss["profile_id"].attrs = profile_meta["profile_id"]
                if "_FillValue" not in dss["profile_id"].attrs:
                    dss["profile_id"].attrs["_FillValue"] = -1
                dss["profile_id"].attrs["valid_min"] = np.int32(
                    dss["profile_id"].attrs["valid_min"]
                )
                dss["profile_id"].attrs["valid_max"] = np.int32(
                    dss["profile_id"].attrs["valid_max"]
                )

                dss["profile_time"] = dss.time.mean()
                dss["profile_time"].attrs = profile_meta["profile_time"]
                # remove units so they can be encoded later:
                try:
                    del dss.profile_time.attrs["units"]
                    del dss.profile_time.attrs["calendar"]
                except KeyError:
                    pass
                dss["profile_lon"] = dss.longitude.mean()
                dss["profile_lon"].attrs = profile_meta["profile_lon"]
                dss["profile_lat"] = dss.latitude.mean()
                dss["profile_lat"].attrs = profile_meta["profile_lat"]

                dss["lat"] = dss["latitude"]
                dss["lon"] = dss["longitude"]
                dss["platform"] = np.int32(1)
                comment = f"{meta['glider_model']} operated by {meta['institution']}"
                dss["platform"].attrs["comment"] = comment
                dss["platform"].attrs["id"] = meta["glider_name"]
                dss["platform"].attrs["instrument"] = instrument_str
                dss["platform"].attrs["long_name"] = (
                    f"{meta['glider_model']} {dss['platform'].attrs['id']}"
                )
                dss["platform"].attrs["type"] = "platform"
                dss["platform"].attrs["wmo_id"] = meta["wmo_id"]
                if "_FillValue" not in dss["platform"].attrs:
                    dss["platform"].attrs["_FillValue"] = -1

                dss["lat_uv"] = np.nan
                dss["lat_uv"].attrs = profile_meta["lat_uv"]
                dss["lon_uv"] = np.nan
                dss["lon_uv"].attrs = profile_meta["lon_uv"]
                dss["time_uv"] = np.nan
                dss["time_uv"].attrs = profile_meta["time_uv"]

                # dss['instrument_ctd'] = np.int32(1.0)
                # dss['instrument_ctd'].attrs = profile_meta['instrument_ctd']
                # if '_FillValue' not in dss['instrument_ctd'].attrs:
                #     dss['instrument_ctd'].attrs['_FillValue'] = -1
                for key in instrument_meta.keys():
                    dss[key] = np.int32(1.0)
                    dss[key].attrs = instrument_meta[key]
                    if "_FillValue" not in dss[key].attrs:
                        dss[key].attrs["_FillValue"] = -1

                dss.attrs["date_modified"] = str(np.datetime64("now")) + "Z"

                # ancillary variables: link and create with values of 2.  If
                # we dont' want them all 2, then create these variables in the
                # time series
                to_fill = [
                    "temperature",
                    "pressure",
                    "conductivity",
                    "salinity",
                    "density",
                    "lon",
                    "lat",
                    "depth",
                ]
                for name in to_fill:
                    qcname = name + "_qc"
                    dss[name].attrs["ancillary_variables"] = qcname
                    if qcname not in dss.keys():
                        dss[qcname] = ("time", 2 * np.ones(len(dss[name]), np.int8))
                        dss[qcname].attrs = pgutils.fill_required_qcattrs({}, name)
                        # 2 is "not eval"

                _log.info("Writing %s", outname)
                timeunits = "seconds since 1970-01-01T00:00:00Z"
                timecalendar = "gregorian"
                try:
                    del dss.profile_time.attrs["_FillValue"]
                    del dss.profile_time.attrs["units"]
                except KeyError:
                    pass
                dss.to_netcdf(
                    outname,
                    encoding={
                        "time": {
                            "units": timeunits,
                            "calendar": timecalendar,
                            "dtype": "float64",
                        },
                        "profile_time": {
                            "units": timeunits,
                            "_FillValue": -99999.0,
                            "dtype": "float64",
                        },
                    },
                )

                # add traj_strlen using bare ntcdf to make IOOS happy
                with netCDF4.Dataset(outname, "r+") as nc:
                    nc.renameDimension("string%d" % trajlen, "traj_strlen")
