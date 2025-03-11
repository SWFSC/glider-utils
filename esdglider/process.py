import os
import re
import logging
import numpy as np
import xarray as xr
import yaml
import subprocess

import esdglider.gcp as gcp
import esdglider.pathutils as pathutils
import esdglider.esdpyglider as esdpyglider
import esdglider.utils as utils

import pyglider.slocum as slocum
import pyglider.ncprocess as ncprocess

_log = logging.getLogger(__name__)


def binary_to_nc(
    deployment, project, mode, deployments_path, config_path, 
    write_timeseries=False, write_gridded=False, write_profiles=False, 
    write_imagery=False, imagery_path=None, 
    min_dt='2017-01-01', profile_force = False
): 
                
    """
    Process raw ESD glider data...

    The contents of this function used to just be in scripts/binary_to_nc.py.
    They were moved to this structure for easier development and debugging

    Parameters
    ----------
    deployment :
    ...
    min_dt : see utils.drop_bogus
    #  profile_filt_time=150, profile_min_time=300, maxgap=300


    Returns
    ----------
    A tuple of the filenames of the various netCDF files. 
    In order: the engineering and science timeseries, 
    and the 1m and 5m gridded files

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
    # deploymentyaml = paths["deploymentyaml"]
    deploymentyaml = os.path.join(config_path, 'deployment-config', 
        f"{deployment}-{mode}.yml")

    #--------------------------------------------
    # TODO: handle compressed files, if necessary. 
    # Although maybe this should be in another function?

    #--------------------------------------------
    # Timeseries
    if write_timeseries:
        if not os.path.exists(tsdir):
            _log.info(f'Creating directory at: {tsdir}')
            os.makedirs(tsdir)
        
        if not os.path.isfile(deploymentyaml):
            raise FileNotFoundError(f'Could not find {deploymentyaml}')

        # Engineering
        _log.info(f'Generating engineering timeseries')
        outname_tseng = slocum.binary_to_timeseries(
            paths["binarydir"], paths["cacdir"], tsdir, 
            [deploymentyaml, paths["engyaml"]], 
            search=binary_search, 
            fnamesuffix=f"-{mode}-eng", 
            time_base="m_depth", 
            profile_filt_time = None)
            # profile_filt_time=profile_filt_time, 
            # profile_min_time=profile_min_time, maxgap=maxgap)

        tseng = xr.load_dataset(outname_tseng)
        tseng = postproc_eng_timeseries(tseng, min_dt=min_dt)
        tseng.to_netcdf(outname_tseng)
        _log.info(f'Finished eng timeseries postproc: {outname_tseng}')

        # Science
        # Note - uses sci_water_temp as time_base sensor
        # TODO: make script to check that we aren't shooting ourselves in foot using CTD as base
        _log.info(f'Generating science timeseries')
        outname_tssci = slocum.binary_to_timeseries(
            paths["binarydir"], paths["cacdir"], tsdir, 
            deploymentyaml, 
            search=binary_search, 
            fnamesuffix=f"-{mode}-sci", 
            time_base='sci_water_temp', 
            profile_filt_time = None)
            # profile_filt_time=profile_filt_time, 
            # profile_min_time=profile_min_time, maxgap=maxgap)

        tssci = xr.load_dataset(outname_tssci)
        tssci = postproc_sci_timeseries(tssci, min_dt=min_dt)
        tssci.to_netcdf(outname_tssci)
        _log.info(f'Finished sci timeseries postproc: {outname_tssci}')

        num_profiles_eng = len(np.unique(tseng.profile_index.values))
        num_profiles_sci = len(np.unique(tssci.profile_index.values))
        if num_profiles_eng != num_profiles_sci: 
            _log.warning("The eng and sci timeseries have different total numbers of profiles")
            _log.debug(f"Number of eng profiles: {num_profiles_eng}")
            _log.debug(f"Number of sci profiles: {num_profiles_sci}")

    else:
        _log.info(f'Not writing timeseries')
        # Get deployment and thus file name from yaml file
        with open(deploymentyaml) as fin:
            deployment_ = yaml.safe_load(fin)
            deployment_name = deployment_["metadata"]["deployment_name"]
        
        outname_tseng = os.path.join(tsdir, f"{deployment_name}-{mode}-eng.nc")
        outname_tssci = os.path.join(tsdir, f"{deployment_name}-{mode}-sci.nc")
        # if not os.path.isfile(outname_tssci):
        #     raise FileNotFoundError(f'Not writing timeseries, and could not find {outname_tssci}')
        # _log.info(f'Reading in outname_tssci ({outname_tssci})')

    #--------------------------------------------
    # TODO: Profiles
    if write_profiles:
        _log.info(f'Generating profile nc files')
        esdpyglider.esd_extract_timeseries_profiles(
            outname_tssci, paths["profdir"], deploymentyaml, profile_force=False)
    else:
        _log.info(f'Not writing profiles')


    #--------------------------------------------
    # Gridded data, 1m and 5m
    # TODO: filter to match SOCIB?
    if write_gridded:
        if not os.path.isfile(outname_tssci):
            raise FileNotFoundError(f'Could not find {outname_tssci}')

        _log.info(f'Generating 1m gridded data')
        outname_1m = ncprocess.make_gridfiles(
            outname_tssci, paths["griddir"], deploymentyaml, 
            dz = 1, fnamesuffix=f"-{mode}-1m")
        _log.info(f'Finished making 1m gridded data: {outname_1m}')

        _log.info(f'Generating 5m gridded data')
        outname_5m = ncprocess.make_gridfiles(
            outname_tssci, paths["griddir"], deploymentyaml, 
            dz = 5, fnamesuffix=f"-{mode}-5m")
        _log.info(f'Finished making 5m gridded data: {outname_5m}')

    else:
        outname_1m = ""
        outname_5m = ""
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
    return outname_tseng, outname_tssci, outname_1m, outname_5m


def postproc_eng_timeseries(ds, min_dt='2017-01-01'):
    """
    Post-process engineering timeseries, including: 
        - Removing CTD vars
        - Calculating profiles using depth (m_depth)
        - Updating attributes

    ds : `xarray.Dataset`
        engineering Dataset, usually passed from binary_to_nc.py
    min_dt: see drop_bogus_times
    
    returns Dataset
    """

    _log.debug(f"begin eng postproc: ds has {len(ds.time)} values")

    # Drop CTD variables required or created by binary_to_timeseries
    ds = ds.drop_vars([
        "depth", "conductivity", "temperature", "pressure", "salinity", 
        "potential_density", "density", "potential_temperature"])
    
    # With depth (CTD) gone, rename depth_measured
    ds = ds.rename({"depth_measured": "depth"})
    
    # Remove times < min_dt
    ds = utils.drop_bogus(ds, "eng", min_dt)

    # Calculate profiles using measured depth
    if np.any(np.isnan(ds.depth.values)):
        num_nan = sum(np.isnan(ds.depth.values))
        _log.warning(f"There are {num_nan} nan depth values")
    ds = utils.get_fill_profiles(ds, ds.time.values, ds.depth.values)

    # Add comment
    if not ('comment' in ds.attrs): 
        ds.attrs["comment"] = "engineering-only time series"
    elif not ds.attrs["comment"].strip():
        ds.attrs["comment"] = "engineering-only time series"
    else:
        ds.attrs["comment"] = ds.attrs["comment"] + "; engineering-only time series"

    _log.debug(f"end eng postproc: ds has {len(ds.time)} values")

    return ds


def postproc_sci_timeseries(ds, min_dt='2017-01-01'):
    """
    Post-process science timeseries, including: 
        - remove bogus times. Eg, 1970 or before deployment start date
        - Calculating profiles using depth (derived from ctd's pressure)

    ds : `xarray.Dataset`
        science Dataset, usually passed from binary_to_nc.py
    min_dt: see drop_bogus_times
    
    returns Dataset
    """

    _log.debug(f"begin sci postproc: ds has {len(ds.time)} values")

    # Remove times < min_dt
    ds = utils.drop_bogus(ds, "sci", min_dt)

    # Calculate profiles, using the CTD-derived depth values
    ds = utils.get_fill_profiles(ds, ds.time.values, ds.depth.values)
    # TODO: update this to play nice with eng timeseries for rt data?

    _log.debug(f"end sci postproc: ds has {len(ds.time)} values")

    return ds


def scrape_sfmc(deployment, project, bucket, sfmc_path, gcpproject_id, secret_id):
    """
    rsync files from sfmc, and send them to correct bucket directories;
    Returns 0
    """

    # logging.basicConfig(
    #     format='%(module)s:%(levelname)s:%(message)s [line %(lineno)d]', 
    #     level=getattr(logging, args.loglevel.upper()))
    
    # #--------------------------------------------
    # # Get args variables
    # deployment = args.deployment
    # project = args.project

    # sfmc_path = args.sfmc_path
    # # sfmc_pwd_file_name = args.sfmc_pwd_file_name
    # gcpproject_id = args.gcpproject_id
    # bucket = args.bucket
    # secret_id = args.secret_id

    _log.info(f'Scraping files from SFMC for deployment {deployment}')


    #--------------------------------------------
    # Checks
    deployment_split = deployment.split('-')
    if len(deployment_split[1]) != 8:
        _log.error('The deployment must be the glider name followed by the deployment date')
        raise ValueError ("Unsuccessful deployment_split")
    else:
        glider = deployment_split[0]
        year = pathutils.year_path(project, deployment)

    #--------------------------------------------
    # Create sfmc directory structure, if needed
    _log.info(f'Making sfmc deployment dirs at {sfmc_path}')
    sfmc_local_path = os.path.join(sfmc_path, f'sfmc-{deployment}/')
    pathutils.mkdir_pass(sfmc_path)
    pathutils.mkdir_pass(sfmc_local_path)

    # sfmc_pwd_file = os.path.join(sfmc_local_path, ".sfmcpwd.txt")
    # _log.debug(f'SFMC ssh password written to {sfmc_pwd_file}')
    # if not os.path.isfile(sfmc_pwd_file):
    #     _log.info('Writing SFMC ssh pwd to file')
    #     file = open(sfmc_pwd_file, 'w+')
    #     file.write(gcp.access_secret_version(gcpproject_id, secret_id))
    #     file.close()
    #     os.chmod(sfmc_pwd_file, stat.S_IREAD)

    #--------------------------------------------
    # rsync with SFMC
    _log.info(f'Starting rsync with SFMC dockerver for {glider}')
    sfmc_glider = os.path.join(
        '/var/opt/sfmc-dockserver/stations/noaa/gliders', 
        glider, 'from-glider/')
    sfmc_server_path = f'swoodman@sfmc.webbresearch.com:{sfmc_glider}'

    rsync_args = [
        'sshpass', '-p', gcp.access_secret_version(gcpproject_id, secret_id), 
        'rsync', "-aP", "--delete", sfmc_server_path, sfmc_local_path]
    # NOTE: sshpass via file does not currently work. Unsure why
    # rsync_args = ['sshpass', '-f', sfmc_pwd_file, 
    #               'rsync', "-aP", "--delete", sfmc_server_path, sfmc_local_path]
    # os.remove(sfmc_pwd_file) #delete sfmc_pwd_file
    # _log.debug(f'Removed SFMC ssh password file')

    _log.debug(rsync_args)
    retcode = subprocess.run(rsync_args, capture_output=True)
    _log.debug(retcode.args)

    if retcode.returncode != 0:
        _log.error('Error rsyncing with SFMC dockserver')
        _log.error(f'Args: {retcode.args}')
        _log.error(f'stderr: {retcode.stderr}')
        raise ValueError ("Unsuccessful rsync with SFMC dockserver")
    else:
        _log.info(f'Successfully completed rsync with SFMC dockerver for {glider}')
        _log.debug(f'Args: {retcode.args}')
        _log.debug(f'stderr: {retcode.stdout}')

    # Check for unexpected file extensions
    sfmc_file_ext = pathutils.find_extensions(sfmc_local_path)
    file_ext_expected = {".cac", ".CAC", ".sbd", ".tbd", ".ad2"} 
                        #  ".ccc", ".scd", ".tcd", ".cam"
    file_ext_weird = sfmc_file_ext.difference(file_ext_expected)
    if len(file_ext_weird) > 0:
        x = os.listdir(sfmc_local_path)
        logging.warning(f'File with the following extensions ({file_ext_weird}) ' + 
            'were downloaded from the SFMC, ' + 
            'but will not be organized copied to the GCP bucket')
        # logging.warning(f'File list: TODO')


    #--------------------------------------------
    # Copy files to subfolders, and rsync with bucket
    logging.info('Starting file management on VM')
    bucket_deployment = f"gs://{bucket}/{project}/{year}/{deployment}"

    # cache files
    name_cac  = 'cac'
    pathutils.mkdir_pass(os.path.join(sfmc_local_path, name_cac))
    rt_file_mgmt(
        sfmc_file_ext, '.[Cc][Aa][Cc]', name_cac, sfmc_local_path, 
        f'gs://{bucket}/cache', rsync_delete=False)

    # sbd/tbd files
    name_stbd = 'stbd'
    bucket_stbd = os.path.join(bucket_deployment, 'data', 'binary', 'rt')
    pathutils.mkdir_pass(os.path.join(sfmc_local_path, name_stbd))
    rt_file_mgmt(
        sfmc_file_ext, '.[SsTt]bd', name_stbd, sfmc_local_path, bucket_stbd)

    # Do not bother with compressed files, because the SFMC uncompresses them
    
    # name_ccc  = 'ccc'
    # pathutils.mkdir_pass(os.path.join(sfmc_local_path, name_ccc))
    # rt_file_mgmt(sfmc_file_ext, '.ccc', name_ccc, sfmc_local_path, 
    #              f'gs://{bucket}/cache-compressed')

    # # scd/tcd files
    # name_stcd = 'stcd'
    # bucket_stcd = os.path.join(bucket_deployment, 'data', 'binary', 'rt-compressed')
    # pathutils.mkdir_pass(os.path.join(sfmc_local_path, name_stcd))
    # rt_file_mgmt(
    #     sfmc_file_ext, '.[SsTt]cd', name_stcd, sfmc_local_path, bucket_stcd)

    # ad2 files
    name_ad2  = 'ad2'
    bucket_ad2 = f"gs://amlr-gliders-acoustics-dev/{project}/{year}/{deployment}/data/rt/"
    pathutils.mkdir_pass(os.path.join(sfmc_local_path, name_ad2))
    rt_file_mgmt(sfmc_file_ext, '.ad2', name_ad2, sfmc_local_path, bucket_ad2)

    # # cam files TODO
    # name_cam  = 'cam'    
    # pathutils.mkdir_pass(os.path.join(sfmc_local_path, name_cam))
    # rt_files_mgmt(sfmc_file_ext, '.cam', name_cam, sfmc_local_path, bucket_cam)

    #--------------------------------------------
    return 0


def rt_file_mgmt(
    sfmc_ext_all, ext_regex, subdir_name, local_path, bucket_path, 
    rsync_delete = True
):
    """
    Move real-time files from the local sfmc folder (local_path)
    to their subdirectory (subdir_path). 
    Then uses gcloud to rsync to their place in the bucket (bucket_path)

    The rsync_delete flag indicates if the --delete-unmatched-destination-objects
    flag is used in the command

    ext_regex_path does include * for copying files (eg is '.[st]bd')
    """
    
    if (any(re.search(ext_regex, i) for i in sfmc_ext_all)):
        # Check paths
        if not os.path.isdir(local_path):
            _log.error(f'Necessary path ({local_path}) does not exist')
            raise FileNotFoundError(f'Could not find {local_path}')

        subdir_path = os.path.join(local_path, subdir_name)
        if not os.path.isdir(subdir_path):
            _log.error(f'Necessary path ({subdir_path}) does not exist')
            raise FileNotFoundError(f'Could not find {subdir_path}')

        # Move files so as to do rsync later
        _log.info(f'Moving {subdir_name} files to their local subdirectory')
        mv_cmd = f'mv {os.path.join(local_path, f'*{ext_regex}')} {subdir_path}'
        _log.debug(mv_cmd)
        retcode_tmp = subprocess.call(mv_cmd, shell = True)
        _log.debug(retcode_tmp)

        # Do rsync
        _log.info(f'Rsyncing {subdir_name} subdirectory with bucket directory')
        rsync_args = ['gcloud', 'storage', 'rsync', '-r']
        if rsync_delete: 
            rsync_args.append("--delete-unmatched-destination-objects")
        rsync_args.extend([subdir_path, bucket_path])

        _log.debug(rsync_args)
        retcode = subprocess.run(rsync_args, capture_output = True)

        if retcode.returncode != 0:
            _log.error(f'Error rsyncing {subdir_name} files to bucket')
            _log.error(f'Args: {retcode.args}')
            _log.error(f'stderr: {retcode.stderr}')
            raise ValueError ("Unsuccessful rsync to bucket")
        else:
            _log.info(f'Rsynced {subdir_name} files to {bucket_path}')
            _log.debug(f'Args: {retcode.args}')
            _log.debug(f'stderr: {retcode.stdout}')
    else: 
        _log.info(f'No {subdir_name} files to copy')

    return 0

