#!/usr/bin/env python

import sys
import logging
import argparse
import esdglider.process as process


def main(args):
    """
    Process binary ESD glider data...
    """

    loglevel = args.loglevel
    log_level = getattr(logging, loglevel.upper())
    log_format = '%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    process.binary_to_nc(
        args.deployment, args.project, args.mode, args.deployments_path, 
        write_timeseries=args.write_timeseries, 
        write_gridded=args.write_gridded, 
        profile_filt_time=args.profile_filt_time, 
        profile_min_time=args.profile_min_time, 
        maxgap=args.maxgap)


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