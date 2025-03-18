import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
import glob
import yaml
import netCDF4
import importlib
from datetime import datetime, timezone

import esdglider.pathutils as pathutils
import esdglider.utils as utils

import pyglider.slocum as slocum
import pyglider.ncprocess as ncprocess
import pyglider.utils as pgutils

_log = logging.getLogger(__name__)


# For encoding time in netCDF files
encoding_dict = {
            'time': {
                'units': 'seconds since 1970-01-01T00:00:00Z',
                '_FillValue': np.nan,                
                'calendar': 'gregorian',
                'dtype': 'float64',
            }
        }


def binary_to_nc(
    deployment, mode, paths, min_dt='2017-01-01', 
    # deployments_path, config_path, 
    write_timeseries=True, write_gridded=True
):     
                
    """
    Process binary ESD glider data to timeseries and/or gridded netCDF files

    The contents of this function used to just be in scripts/binary_to_nc.py.
    They were moved to this structure for easier development and debugging

    Parameters
    ----------
    deployment : str
        The name of the glider deployment. Eg, amlr01-20210101

    mode : str
        Mode of the glider dat being processed. 
        Must be either 'rt', for real-time, or 'delayed

    paths : dict
        A dictionary of file/directory paths for various processing steps. 
        Intended to be the output of esdglider.pathutils.esd_paths()
        See this function for the expected key/value pairs
    
    min_dt : datetime64, or object that can be converted to datetime64
        See utils.drop_bogus; default is '2017-01-01'.
        All timestamps from before this value will be dropped

    write_timeseries, write_gridded : bool
        Should the timeseries and gridded, respectively, 
        xarray DataSets be both created and written to files?
        Note: if True then already-existing files will be clobbered

    Returns
    ----------
    A tuple of the filenames of the various netCDF files, as strings.
    In order: the engineering and science timeseries, 
    and the 1m and 5m gridded files

    """

    # Choices (delayed, rt) specified in arg input
    if mode == 'delayed':
        binary_search = '*.[D|E|d|e]bd'
    else:
        binary_search = '*.[S|T|s|t]bd'

    #--------------------------------------------
    # Check file and directory paths
    # paths = pathutils.esd_paths(
    #     project, deployment, mode, deployments_path, config_path)
    tsdir = paths["tsdir"]
    deploymentyaml = paths["deploymentyaml"]    

    # Get deployment and thus file name from yaml file
    with open(deploymentyaml) as fin:
        deployment_ = yaml.safe_load(fin)
        deployment_name = deployment_["metadata"]["deployment_name"]
    if deployment_name != deployment:
        raise ValueError (
            f"Provided deployment ({deployment}) is not the same as " +
            f"the deploymentyaml deployment_name ({deployment_name})"
        )

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

        # Engineering - uses m_depth as time base
        _log.info(f'Generating engineering timeseries')
        outname_tseng = slocum.binary_to_timeseries(
            paths["binarydir"], paths["cacdir"], tsdir, 
            [deploymentyaml, paths["engyaml"]], 
            search=binary_search, 
            fnamesuffix=f"-{mode}-eng", 
            time_base="m_depth", 
            profile_filt_time = None)

        _log.info(f'Post-processing engineering timeseries')
        tseng = xr.load_dataset(outname_tseng)
        tseng = postproc_eng_timeseries(tseng, min_dt=min_dt)
        tseng = postproc_attrs(tseng, mode)
        tseng.to_netcdf(outname_tseng, encoding=encoding_dict)
        _log.info(f'Finished eng timeseries postproc: {outname_tseng}')

        # Science - uses sci_water_temp as time_base sensor
        _log.info(f'Generating science timeseries')
        outname_tssci = slocum.binary_to_timeseries(
            paths["binarydir"], paths["cacdir"], tsdir, 
            deploymentyaml, 
            search=binary_search, 
            fnamesuffix=f"-{mode}-sci", 
            time_base='sci_water_temp', 
            profile_filt_time = None)

        _log.info(f'Post-processing science timeseries')
        tssci = xr.load_dataset(outname_tssci)
        tssci = postproc_sci_timeseries(tssci, min_dt=min_dt)
        tssci = postproc_attrs(tssci, mode)
        tssci.to_netcdf(outname_tssci, encoding=encoding_dict)
        _log.info(f'Finished sci timeseries postproc: {outname_tssci}')

        num_profiles_eng = len(np.unique(tseng.profile_index.values))
        num_profiles_sci = len(np.unique(tssci.profile_index.values))
        if num_profiles_eng != num_profiles_sci: 
            _log.warning("The eng and sci timeseries have different total numbers of profiles")
            _log.debug(f"Number of eng profiles: {num_profiles_eng}")
            _log.debug(f"Number of sci profiles: {num_profiles_sci}")

    else:
        _log.info(f'Not writing timeseries')
        outname_tseng = os.path.join(tsdir, f"{deployment}-{mode}-eng.nc")
        outname_tssci = os.path.join(tsdir, f"{deployment}-{mode}-sci.nc")
        
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

        _log.info(f'Generating 5m gridded data')
        outname_5m = ncprocess.make_gridfiles(
            outname_tssci, paths["griddir"], deploymentyaml, 
            dz = 5, fnamesuffix=f"-{mode}-5m")

    else:
        _log.info(f'Not writing gridded data')
        outname_1m = os.path.join(paths["griddir"], f"{deployment}_grid-{mode}-1m.nc")
        outname_5m = os.path.join(paths["griddir"], f"{deployment}_grid-{mode}-5m.nc")

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


def postproc_attrs(ds, mode):
    """
    Update attrbites of xarray DataSet ds
    Used for both engineering and science timeseries
    """
    try:
        del ds.attrs['glider_serial']
    except KeyError:
        _log.warning("Unable to delete glider_serial attribute")
        pass

    ds.attrs['standard_name_vocabulary'] = 'CF Standard Name Table v72'
    ds.attrs['history'] = (        
        f"{np.datetime64('now')}Z: netCDF files created using: " +
        f"pyglider v{importlib.metadata.version("pyglider")}; " + 
        f"esdglider v{importlib.metadata.version("esdglider")}"
    )
    ds.attrs['processing_level'] = (
        "Minimal data screening. " + 
        "Data provided as is, with no expressed or implied assurance " +
        "of quality assurance or quality control."
    )

    if mode == "delayed":
        ds.attrs['title'] = ds.attrs['title'] + "-delayed"

    return ds


def postproc_eng_timeseries(ds, min_dt='2017-01-01'):
    """
    Post-process engineering timeseries, including: 
        - Removing CTD vars
        - Calculating profiles using depth (m_depth)
        - Updating attributes

    ds : `xarray.Dataset`
        engineering Dataset, usually passed from binary_to_nc.py
    min_dt: passed to drop_bogus_times
    
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

    # Reorder data variables
    new_start = ['latitude', 'longitude', 'depth', 'profile_index']
    ds = utils.data_var_reorder(ds, new_start)

    # Update comment
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
        - remove bogus times. Eg, 1970, or before deployment start date
        - Calculating profiles using depth (derived from ctd's pressure)

    ds : `xarray.Dataset`
        science Dataset, usually passed from binary_to_nc.py
    min_dt: passed to drop_bogus_times
    
    returns Dataset
    """

    _log.debug(f"begin sci postproc: ds has {len(ds.time)} values")

    # Remove times < min_dt
    ds = utils.drop_bogus(ds, "sci", min_dt)

    # Calculate profiles, using the CTD-derived depth values
    # TODO: update this to play nice with eng timeseries for rt data?
    ds = utils.get_fill_profiles(ds, ds.time.values, ds.depth.values)

    # Reorder data variables
    new_start = ['latitude', 'longitude', 'depth', 'profile_index', 
                 'conductivity', 'temperature', 'pressure', 'salinity', 
                 'density', 'potential_temperature', 'potential_density']
    ds = utils.data_var_reorder(ds, new_start)

    _log.debug(f"end sci postproc: ds has {len(ds.time)} values")

    return ds


def imagery_timeseries(ds_eng, ds_sci, imagery_dir, ext = 'jpg'):
    """
    Matches up imagery files with data from pyglider by imagery filename
    Uses interpolated variables (hardcoded in function)
    Returns data frame with iamgery+timeseries information
    
    Args:
        ds_eng (Dataset): xarray Dataset; from engineering timeseries NetCDF
        ds_sci (Dataset): xarray Dataset; from science timeseries NetCDF
        imagery_dir (str): path to folder with images, 
            specifically the 'Dir####' folders
        ext (str, optional): Imagery file extension. Default is 'jpg'.

    Returns:
        DataFrame: pd.DataFrame of imagery timeseries
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
        imagery_files_dt = [utils.solocam_filename_dt(i, yr_idx) for i in imagery_files]
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



def ngdac_profiles(inname, outdir, deploymentyaml, force=False):
    """
    ESD's version of extract_timeseries_profiles, from:
    https://github.com/c-proof/pyglider/blob/main/pyglider/ncprocess.py#L19
    
    Extract and save each profile from a timeseries netCDF.

    Parameters
    ----------
    inname : str or Path
        netcdf file to break into profiles

    outdir : str or Path
        directory to place profiles

    deploymentyaml : str or Path
        location of deployment yaml file for the netCDF file.  This should
        be the same yaml file that was used to make the timeseries file.

    force : bool, default False
        Force an overwite even if profile netcdf already exists
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

    meta = deployment['metadata']
    with xr.open_dataset(inname) as ds:
        _log.info('Extracting profiles: opening %s', inname)
        trajectory = utils.esd_file_id(ds).encode()
        trajlen    = len(trajectory)
        
        # TODO: do floor like oceanGNS??
        profiles = np.unique(ds.profile_index)
        profiles = [p for p in profiles if (~np.isnan(p) and not (p % 1) and (p > 0))]
        for p in profiles:
            ind = np.where(ds.profile_index == p)[0]
            dss = ds.isel(time=ind)
            outname = outdir + '/' + utils.esd_file_id(dss) + '.nc'
            _log.info('Checking %s', outname)
            if force or (not os.path.exists(outname)):
                # this is the id for the whole file, not just this profile..
                dss['trajectory'] = trajectory
                # trajlen = len(pgutils.get_file_id(ds).encode())
                dss['trajectory'].attrs['cf_role'] = 'trajectory_id'
                dss['trajectory'].attrs['comment'] = (
                    'A trajectory is a single'
                    'deployment of a glider and may span multiple data files.'
                )
                dss['trajectory'].attrs['long_name'] = 'Trajectory/Deployment Name'

                # profile-averaged variables....
                profile_meta = deployment['profile_variables']
                if 'water_velocity_eastward' in dss.keys():
                    dss['u'] = dss.water_velocity_eastward.mean()
                    dss['u'].attrs = profile_meta['u']

                    dss['v'] = dss.water_velocity_northward.mean()
                    dss['v'].attrs = profile_meta['v']
                elif 'u' in profile_meta:
                    dss['u'] = profile_meta['u'].get('_FillValue', np.nan)
                    dss['u'].attrs = profile_meta['u']

                    dss['v'] = profile_meta['v'].get('_FillValue', np.nan)
                    dss['v'].attrs = profile_meta['v']
                else:
                    dss['u'] = np.nan
                    dss['v'] = np.nan

                dss['profile_id'] = np.int32(p)
                dss['profile_id'].attrs = profile_meta['profile_id']
                if '_FillValue' not in dss['profile_id'].attrs:
                    dss['profile_id'].attrs['_FillValue'] = -1
                dss['profile_id'].attrs['valid_min'] = np.int32(
                    dss['profile_id'].attrs['valid_min']
                )
                dss['profile_id'].attrs['valid_max'] = np.int32(
                    dss['profile_id'].attrs['valid_max']
                )

                dss['profile_time'] = dss.time.mean()
                dss['profile_time'].attrs = profile_meta['profile_time']
                # remove units so they can be encoded later:
                try:
                    del dss.profile_time.attrs['units']
                    del dss.profile_time.attrs['calendar']
                except KeyError:
                    pass
                dss['profile_lon'] = dss.longitude.mean()
                dss['profile_lon'].attrs = profile_meta['profile_lon']
                dss['profile_lat'] = dss.latitude.mean()
                dss['profile_lat'].attrs = profile_meta['profile_lat']

                dss['lat'] = dss['latitude']
                dss['lon'] = dss['longitude']
                dss['platform'] = np.int32(1)
                comment = f"{meta['glider_model']} operated by {meta['institution']}"
                dss['platform'].attrs['comment'] = comment
                dss['platform'].attrs['id'] = meta['glider_name']
                dss['platform'].attrs['instrument'] = instrument_str
                dss['platform'].attrs['long_name'] = (
                    f"{meta['glider_model']} {dss['platform'].attrs['id']}")
                dss['platform'].attrs['type'] = 'platform'
                dss['platform'].attrs['wmo_id'] = meta['wmo_id']
                if '_FillValue' not in dss['platform'].attrs:
                    dss['platform'].attrs['_FillValue'] = -1

                dss['lat_uv'] = np.nan
                dss['lat_uv'].attrs = profile_meta['lat_uv']
                dss['lon_uv'] = np.nan
                dss['lon_uv'].attrs = profile_meta['lon_uv']
                dss['time_uv'] = np.nan
                dss['time_uv'].attrs = profile_meta['time_uv']

                # dss['instrument_ctd'] = np.int32(1.0)
                # dss['instrument_ctd'].attrs = profile_meta['instrument_ctd']
                # if '_FillValue' not in dss['instrument_ctd'].attrs:
                #     dss['instrument_ctd'].attrs['_FillValue'] = -1
                for key in instrument_meta.keys():
                    dss[key] = np.int32(1.0)
                    dss[key].attrs = instrument_meta[key]
                    if '_FillValue' not in dss[key].attrs:
                        dss[key].attrs['_FillValue'] = -1

                dss.attrs['date_modified'] = str(np.datetime64('now')) + 'Z'

                # ancillary variables: link and create with values of 2.  If
                # we dont' want them all 2, then create these variables in the
                # time series
                to_fill = [
                    'temperature',
                    'pressure',
                    'conductivity',
                    'salinity',
                    'density',
                    'lon',
                    'lat',
                    'depth',
                ]
                for name in to_fill:
                    qcname = name + '_qc'
                    dss[name].attrs['ancillary_variables'] = qcname
                    if qcname not in dss.keys():
                        dss[qcname] = ('time', 2 * np.ones(len(dss[name]), np.int8))
                        dss[qcname].attrs = pgutils.fill_required_qcattrs({}, name)
                        # 2 is "not eval"

                _log.info('Writing %s', outname)
                timeunits = 'seconds since 1970-01-01T00:00:00Z'
                timecalendar = 'gregorian'
                try:
                    del dss.profile_time.attrs['_FillValue']
                    del dss.profile_time.attrs['units']
                except KeyError:
                    pass
                dss.to_netcdf(
                    outname,
                    encoding={
                        'time': {
                            'units': timeunits,
                            'calendar': timecalendar,
                            'dtype': 'float64',
                        },
                        'profile_time': {
                            'units': timeunits,
                            '_FillValue': -99999.0,
                            'dtype': 'float64',
                        },
                    },
                )

                # add traj_strlen using bare ntcdf to make IOOS happy
                with netCDF4.Dataset(outname, 'r+') as nc:
                    nc.renameDimension('string%d' % trajlen, 'traj_strlen')
