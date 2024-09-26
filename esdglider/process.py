import os
import logging
import numpy as np
import xarray as xr
import yaml

import esdglider.pathutils as pathutils
import esdglider.utils as utils

import pyglider.slocum as slocum
import pyglider.ncprocess as ncprocess

_log = logging.getLogger(__name__)


def binary_to_nc(deployment, project, mode, deployments_path, 
                 write_timeseries=False, write_gridded=False, 
                 write_imagery=False, imagery_path=None, 
                 profile_filt_time=100, profile_min_time=300, maxgap=300):
    """
    Process raw ESD glider data...

    The contents of this function used to just be in scripts/binary_to_nc.py.
    They were moved to this structure for easier development and debugging
    """

    # Choices (delayed, rt) specified in arg input
    if mode == 'delayed':
        binary_search = '*.[D|E|d|e]bd'
    else:
        binary_search = '*.[S|T|s|t]bd'

    #--------------------------------------------
    # Check/make file and directory paths
    paths = pathutils.esd_paths(project, deployment, mode, deployments_path)
    tsdir = paths["tsdir"]

    #--------------------------------------------
    # TODO: handle compressed files, if necessary

    #--------------------------------------------
    # Timeseries
    if write_timeseries:
        _log.info(f'Writing timeseries')
        if not os.path.exists(tsdir):
            _log.info(f'Creating directory at: {tsdir}')
            os.makedirs(tsdir)

        # Engineering
        _log.info(f'Generating engineering timeseries')
        outname_tseng = slocum.binary_to_timeseries(
            paths["binarydir"], paths["cacdir"], tsdir, 
            [paths["deploymentyaml"], paths["engyaml"]], 
            search=binary_search, fnamesuffix='-eng', time_base='m_depth',
            profile_filt_time=profile_filt_time, 
            profile_min_time=profile_min_time, maxgap=maxgap)

        tseng = xr.open_dataset(outname_tseng)
        tseng = postproc_eng_timeseries(tseng)
        tseng.close()
        tseng.to_netcdf(outname_tseng)
        _log.info(f'Finished eng timeseries postproc: {outname_tseng}')

        # Science
        # TODO: make script to check that we aren't shooting ourselves in foot using CTD as base
        _log.info(f'Generating science timeseries')
        outname_tssci = slocum.binary_to_timeseries(
            paths["binarydir"], paths["cacdir"], tsdir, 
            paths["deploymentyaml"],
            search=binary_search, fnamesuffix='-sci',  time_base='sci_water_temp',
            profile_filt_time=profile_filt_time, 
            profile_min_time=profile_min_time, maxgap=maxgap)

        tssci = xr.open_dataset(outname_tssci)
        tssci = postproc_sci_timeseries(tssci)
        tssci.close()
        tssci.to_netcdf(outname_tssci)
        _log.info(f'Finished sci timeseries postproc: {outname_tssci}')

    else:
        _log.info(f'Not writing timeseries')
        # Get deployment and thus file name from yaml file
        with open(paths["deploymentyaml"]) as fin:
            deployment_ = yaml.safe_load(fin)
            deployment_name = deployment_["metadata"]["deployment_name"]
        
        # outname_tseng = os.path.join(tsdir, deployment, '-eng.nc')
        outname_tssci = os.path.join(tsdir, f"{deployment_name}-sci.nc")
        if not os.path.isfile(outname_tssci):
            raise FileNotFoundError(f'Not writing timeseries, and could not find {outname_tssci}')
        _log.info(f'Reading in outname_tssci ({outname_tssci})')

    #--------------------------------------------
    # TODO: Profiles
    # extract_timeseries_profiles(inname, outdir, deploymentyaml, force=False)

    #--------------------------------------------
    # Gridded data, 1m and 5m
    # TODO: filter to match SOCIB?
    if write_gridded:
        _log.info(f'Generating 1m gridded data')
        outname_1m = ncprocess.make_gridfiles(
            outname_tssci, paths["griddir"], paths["deploymentyaml"], 
            dz = 1, fnamesuffix="-1m")
        _log.info(f'Finished making 1m gridded data: {outname_1m}')

        _log.info(f'Generating 5m gridded data')
        outname_5m = ncprocess.make_gridfiles(
            outname_tssci, paths["griddir"], paths["deploymentyaml"], 
            dz = 5, fnamesuffix="-5m")
        _log.info(f'Finished making 5m gridded data: {outname_5m}')

    else:
        _log.info(f'Not writing gridded data')

    #--------------------------------------------
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

    #--------------------------------------------
    return 0


def postproc_eng_timeseries(ds, min_dt='1971-01-01'):
    """
    Post-process engineering timeseries, including: 
        - Removing CTD vars
        - Calculating profiles using m_depth instead of pressure
        - Updating attributes

    ds : `xarray.Dataset`
        engineering Dataset, usually passed from binary_to_nc.py
    min_dt: see drop_bogus_times
    
    returns Dataset
    """

    # Drop CTD variables required by binary_to_timeseries
    ds = ds.drop_vars(["depth", "conductivity", "temperature", "pressure", 
                       "salinity", "potential_density", "density", 
                       "potential_temperature"])
    
    # With depth gone, rename m_depth
    ds = ds.rename({"m_depth": "depth"})

    # Remove times < min_dt
    ds = utils.drop_bogus_times(ds, min_dt)

    # Calculate profile indices using measured depth
    if np.any(np.isnan(ds.depth.values)):
        num_nan = sum(np.isnan(ds.depth.values))
        _log.warning(f"There are {num_nan} nan depth values")
    ds = utils.get_profiles_esd(ds, "depth")
    _log.debug(f"There are {np.max(ds.profile_index.values)} profiles")

    # Add comment
    if not ('comment' in ds.attrs): 
        ds.attrs["comment"] = "engineering-only time series"
    elif not ds.attrs["comment"].strip():
        ds.attrs["comment"] = "engineering-only time series"
    else:
        ds.attrs["comment"] = ds.attrs["comment"] + "; engineering-only time series"

    return ds


def postproc_sci_timeseries(ds, min_dt='1971-01-01'):
    """
    Post-process science timeseries, including: 
        - rename to depth_ctd and depth
        - remove bogus times. Eg, 1970 or before deployment start date
        - Profiles. How to calc when ctd only sampling on dives?
        - 

    ds : `xarray.Dataset`
        science Dataset, usually passed from binary_to_nc.py
    min_dt: see drop_bogus_times
    
    returns Dataset
    """

    # Rename variables
    ds = ds.rename({"depth": "depth_ctd", 
                    "m_depth": "depth"})
    
    # Remove times < min_dt
    ds = utils.drop_bogus_times(ds, min_dt)

    return ds