#!/usr/bin/env python

import os
import sys
import logging
import argparse
import xarray as xr
import yaml

from esdglider.paths import esd_paths
from esdglider.esdutils import postproc_eng_timeseries, postproc_sci_timeseries

import pyglider.slocum as slocum
import pyglider.ncprocess as ncprocess


def main(args):
    """
    Process raw ESD glider data...
    """

    loglevel = args.loglevel
    log_level = getattr(logging, loglevel.upper())
    log_format = '%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    deployment = args.deployment
    project = args.project
    mode = args.mode
    deployments_path = args.deployments_path
    # clobber = args.clobber

    # Choices (delayed, rt) specified in arg input
    if mode == 'delayed':
        binary_search = '*.[D|E|d|e]bd'
    else:
        binary_search = '*.[S|T|s|t]bd'

    # write_trajectory = args.write_trajectory
    # write_ngdac = args.write_ngdac    
    # write_acoustics = args.write_acoustics
    # write_imagery = args.write_imagery
    # imagery_path = args.imagery_path

    #--------------------------------------------
    # Check/make file and directory paths
    paths = esd_paths(project, deployment, mode, deployments_path)
    tsdir = paths["tsdir"]

    #--------------------------------------------
    # TODO: handle compressed files, if necessary

    #--------------------------------------------
    # Timeseries
    if args.write_timeseries:
        logging.info(f'Writing timeseries')
        if not os.path.exists(tsdir):
            logging.info(f'Creating directory at: {tsdir}')
            os.makedirs(tsdir)

        # Engineering
        logging.info(f'Generating engineering timeseries')
        outname_tseng = slocum.binary_to_timeseries(
            paths["binarydir"], paths["cacdir"], tsdir, 
            [paths["deploymentyaml"], paths["engyaml"]], 
            search=binary_search, fnamesuffix='-eng', time_base='m_depth',
            profile_filt_time=args.profile_filt_time, 
            profile_min_time=args.profile_min_time, maxgap=args.maxgap)

        tseng = xr.open_dataset(outname_tseng)
        tseng = postproc_eng_timeseries(tseng)
        tseng.close()
        tseng.to_netcdf(outname_tseng)
        logging.info(f'Finished eng timeseries postproc: {outname_tseng}')

        # Science
        # TODO: make script to check that we aren't shooting ourselves in foot using CTD as base
        logging.info(f'Generating science timeseries')
        outname_tssci = slocum.binary_to_timeseries(
            paths["binarydir"], paths["cacdir"], tsdir, 
            paths["deploymentyaml"],
            search=binary_search, fnamesuffix='-sci',  time_base='sci_water_temp',
            profile_filt_time=args.profile_filt_time, 
            profile_min_time=args.profile_min_time, maxgap=args.maxgap)

        tssci = xr.open_dataset(outname_tssci)
        tssci = postproc_sci_timeseries(tssci)
        tssci.close()
        tssci.to_netcdf(outname_tssci)
        logging.info(f'Finished sci timeseries postproc: {outname_tssci}')

    else:
        logging.info(f'Not writing timeseries')
        # Get deployment and thus file name from yaml file
        with open(paths["deploymentyaml"]) as fin:
            deployment_ = yaml.safe_load(fin)
            deployment_name = deployment_["metadata"]["deployment_name"]
        
        # outname_tseng = os.path.join(tsdir, deployment, '-eng.nc')
        outname_tssci = os.path.join(tsdir, f"{deployment_name}-sci.nc")
        if not os.path.isfile(outname_tssci):
            raise FileNotFoundError(f'Not writing timeseries, and could not find {outname_tssci}')
        logging.info(f'Reading in outname_tssci ({outname_tssci})')

    #--------------------------------------------
    # TODO: Profiles
    # extract_timeseries_profiles(inname, outdir, deploymentyaml, force=False)

    #--------------------------------------------
    # Gridded data, 1m and 5m
    # TODO: filter to match SOCIB?
    if args.write_gridded:
        logging.info(f'Generating 1m gridded data')
        outname_1m = ncprocess.make_gridfiles(
            outname_tssci, paths["griddir"], paths["deploymentyaml"], 
            dz = 1, fnamesuffix="-1m")
        logging.info(f'Finished making 1m gridded data: {outname_1m}')

        logging.info(f'Generating 5m gridded data')
        outname_5m = ncprocess.make_gridfiles(
            outname_tssci, paths["griddir"], paths["deploymentyaml"], 
            dz = 5, fnamesuffix="-5m")
        logging.info(f'Finished making 5m gridded data: {outname_5m}')

    else:
        logging.info(f'Not writing gridded data')

    #--------------------------------------------
    # Write imagery metadata file
    # if write_imagery:
    #     logging.info("write_imagery is True, and thus writing imagery metadata file")
    #     if mode == 'rt':
    #         logging.warning('You are creating imagery file metadata ' + 
    #             'using real-time data. ' + 
    #             'This may result in inaccurate imagery file metadata')
    #     amlr_imagery_metadata(
    #         gdm, deployment, glider_path, 
    #         os.path.join(imagery_path, 'gliders', args.ugh_imagery_year, deployment)
    #     )

    #--------------------------------------------
    return 0



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        allow_abbrev=False)

    arg_parser.add_argument('deployment', 
        type=str,
        help='Deployment name, eg amlr03-20220425')

    arg_parser.add_argument('project', 
        type=str,
        help='Glider project name', 
        choices=['FREEBYRD', 'REFOCUS', 'SANDIEGO'])

    arg_parser.add_argument('mode', 
        type=str,
        help="Specify which binary files will be converted to dbas. " + 
            "'delayed' means [de]bd files will be converted, " + 
            "and 'rt' means [st]bd files will be converted", 
        choices=['delayed', 'rt'])

    arg_parser.add_argument('deployments_path', 
        type=str,
        help='Path to glider deployments directory. ' + 
            'In GCP, this will be the mounted bucket path')

    arg_parser.add_argument('--profile_filt_time', 
        type=float,
        default=100,
        help="time in seconds over which to smooth the pressure " +
            "time series for finding up and down profiles. " +
            "https://github.com/c-proof/pyglider/blob/main/pyglider/slocum.py")
    
    arg_parser.add_argument('--profile_min_time', 
        type=float,
        default=300,
        help="minimum time to consider a profile an actual profile (seconds). " +
            "https://github.com/c-proof/pyglider/blob/main/pyglider/slocum.py")
    
    arg_parser.add_argument('--maxgap', 
        type=float,
        default=300,
        help="Longest gap in seconds to interpolate over " +
            "when matching instrument timeseries. " +
            "https://github.com/c-proof/pyglider/blob/main/pyglider/slocum.py")

    arg_parser.add_argument('--write_timeseries',
        help='flag; indicates if timeseries nc files should be written',
        action='store_true')

    arg_parser.add_argument('--write_gridded',
        help='flag; indicates if gridded nc files should be written',
        action='store_true')

    # arg_parser.add_argument('--write_ngdac',
    #     help='flag; indicates if ngdac nc files should be written',
    #     action='store_true')

    # arg_parser.add_argument('--write_acoustics',
    #     help='flag; indicates if acoustic files should be written',
    #     action='store_true')

    arg_parser.add_argument('--write_imagery',
        help='flag; indicates if imagery metadata csv file should be written',
        action='store_true')

    arg_parser.add_argument('--imagery_path',
        type=str,
        help='Path to imagery bucket. Required if write_imagery flag is true',
        default='')

    arg_parser.add_argument('-l', '--loglevel',
        type=str,
        help='Verbosity level',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info')
    
    # arg_parser.add_argument('--logfile',
    #     type=str,
    #     help='File to which to write logs',
    #     default='')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))