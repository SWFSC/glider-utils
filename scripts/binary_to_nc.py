#!/usr/bin/env python

import os
import sys
import logging
import argparse

from esdglider.paths import year_path, binary_to_nc_paths

import pyglider.slocum as slocum
# import pyglider.ncprocess as ncprocess

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
    clobber = args.clobber

    # Choices (delayed, rt) specified in argument input
    if mode == 'delayed':
        binary_search = '*.[D|E|d|e]bd'
    else:
        binary_search = '*.[D|E|d|e]bd'

    # write_trajectory = args.write_trajectory
    # write_ngdac = args.write_ngdac    
    # write_acoustics = args.write_acoustics
    # write_imagery = args.write_imagery
    # imagery_path = args.imagery_path

    #--------------------------------------------
    # Checks and make glider_path and pyglider variables
    paths = binary_to_nc_paths(project, deployment, mode, deployments_path)
    tsdir = paths["tsdir"]

    #--------------------------------------------
    # Run pyglider to process

    if not os.path.exists(tsdir):
        logging.info(f'Creating directory at: {tsdir}')
        os.makedirs(tsdir)

    logging.info(f'Generating engineering timeseries: {tsdir}')
    outname_tseng = slocum.binary_to_timeseries(
        paths["binarydir"], paths["cacdir"], tsdir, 
        [paths["deploymentyaml"], paths["engyaml"]], 
        search=binary_search, fnamesuffix='-eng',
        time_base='m_depth', profile_filt_time=100,
        profile_min_time=300, maxgap=300)

    logging.info(f'Wrote eng timeseries file: {outname_tseng}')
    
    # outname_tssci = slocum.binary_to_timeseries(
    #     paths["binarydir"], paths["cacdir"], tsdir, 
    #     paths["deploymentyaml"],
    #     search=binary_search, fnamesuffix='-sci',
    #     time_base='sci_water_temp', profile_filt_time=100,
    #     profile_min_time=300, maxgap=300)

   




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
    
    arg_parser.add_argument('--clobber',
        help='flag; should the files be clobbered if they exist',
        action='store_true')

    # arg_parser.add_argument('--write_trajectory',
    #     help='flag; indicates if trajectory nc file should be written',
    #     action='store_true')

    # arg_parser.add_argument('--write_ngdac',
    #     help='flag; indicates if ngdac nc files should be written',
    #     action='store_true')

    # arg_parser.add_argument('--write_acoustics',
    #     help='flag; indicates if acoustic files should be written',
    #     action='store_true')

    # arg_parser.add_argument('--write_imagery',
    #     help='flag; indicates if imagery metadata csv file should be written',
    #     action='store_true')

    # arg_parser.add_argument('--imagery_path',
    #     type=str,
    #     help='Path to imagery bucket. Required if write_imagery flag is true',
    #     default='')

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