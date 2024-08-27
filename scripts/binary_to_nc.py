#!/usr/bin/env python

import os
import sys
import logging
import argparse

from esdgliderutils.paths import year_path

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

    # write_trajectory = args.write_trajectory
    # write_ngdac = args.write_ngdac    
    # write_acoustics = args.write_acoustics
    # write_imagery = args.write_imagery
    # imagery_path = args.imagery_path

    #--------------------------------------------
    # Checks and make glider_path and pyglider variables

    prj_list = ['FREEBYRD', 'REFOCUS', 'SANDIEGO']    
    if not os.path.isdir(deployments_path):
        logging.error(f'deployments_path ({deployments_path}) does not exist')
        return
    else:
        dir_expected = prj_list + ['cache']
        if not all(x in os.listdir(deployments_path) for x in dir_expected):
            logging.error(f"The expected folders ({', '.join(dir_expected)}) " + 
                f'were not found in the provided directory ({deployments_path}). ' + 
                'Did you provide the right path via deployments_path?')
            return 

    deployment_split = deployment.split('-')
    deployment_mode = f'{deployment}-{mode}'
    year = year_path(project, deployment_split)

    glider_path = os.path.join(deployments_path, project, year, deployment)
    if not os.path.isdir(glider_path):
        logging.error(f'glider_path ({glider_path}) does not exist')
        return
    
    # if write_imagery:
    #     if not os.path.isdir(imagery_path):
    #         logging.error('write_imagery is true, and thus imagery_path ' + 
    #                       f'({imagery_path}) must be a valid path')
    #         return

    cacdir = os.path.join(deployments_path, 'cache')
    binarydir = os.path.join(glider_path, 'data', 'binary', mode)
    deploymentyaml = os.path.join(glider_path, 'data', 'data-config', 
        f"{deployment_mode}.yml")

    l1tsdir = os.path.join(glider_path, 'data', 'nc', 'L1-timeseries')
    profiledir = os.path.join(glider_path, 'data', 'nc', 'ngdac', mode)
    l1griddir = os.path.join(glider_path, 'data', 'nc', 'L1-gridded')

    #--------------------------------------------
    # Run pyglider to process

    if not os.path.exists(l1tsdir):
        logging.info(f'Creating directory at: {l1tsdir}')
        os.makedirs(l1tsdir)

    l1ts_outname_sci = slocum.binary_to_timeseries(
        binarydir, cacdir, l1tsdir, deploymentyaml,
        search='*.[d|e]bd', fnamesuffix='-sci',
        # search='*.[D|E]BD', fnamesuffix='',
        time_base='sci_water_temp', profile_filt_time=100,
        profile_min_time=300, maxgap=300)

    l1ts_outname_oxy = slocum.binary_to_timeseries(
        binarydir, cacdir, l1tsdir, deploymentyaml,
        search='*.[d|e]bd', fnamesuffix='-oxy',
        # search='*.[D|E]BD', fnamesuffix='',
        time_base='oxygen_concentration', profile_filt_time=100,
        profile_min_time=300, maxgap=300)

    l1ts_outname_eng = slocum.binary_to_timeseries(
        binarydir, cacdir, l1tsdir, deploymentyaml,
        search='*.[d|e]bd', fnamesuffix='-eng',
        # search='*.[D|E]BD', fnamesuffix='',
        time_base='m_depth', profile_filt_time=100,
        profile_min_time=300, maxgap=300)

    # logging.info(f'Wrote L1 timeseries file: {l1ts_outname}')




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