import os
from pathlib import Path
import logging
from importlib.resources import files, as_file

_log = logging.getLogger(__name__)

"""
Functions that generate or otherwise handle ESD glider-specific 
file path operations
"""

def get_engyml_path():
    """
    Get and return the path to the engineering deployment yaml
    Returns the path, so as to ber able to pass to binary_to_timeseries
    """

    ref = files('esdglider.data') / 'deployment-eng.yml'
    with as_file(ref) as path:
        return str(path)


def find_extensions(dir_path): #,  excluded = ['', '.txt', '.lnk']):
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


def year_path(project, deployment):
    """
    From the glider project and deployment name, 
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
    deployment_split = deployment.split('-')
    year = deployment_split[1][0:4]

    if project == 'FREEBYRD':
        month = deployment_split[1][4:6]
        if int(month) <= 7: 
            year = f'{int(year)}'
        else:
            year = f'{int(year)+1}'

    return year


def esd_paths(project, deployment, mode, deployments_path):
    """
    project
    deployment
    mode
    deployments_path

    Return paths needed by the binary_to_nc script.
    Paths as described here:
    https://swfsc.github.io/glider-lab-manual/content/data-management.html
    """
    prj_list = ['FREEBYRD', 'REFOCUS', 'SANDIEGO', 'ECOSWIM']    
    if not os.path.isdir(deployments_path):
        _log.error(f'deployments_path ({deployments_path}) does not exist')
        return
    else:
        dir_expected = prj_list + ['cache']
        if not all(x in os.listdir(deployments_path) for x in dir_expected):
            _log.error(f"The expected folders ({', '.join(dir_expected)}) " + 
                f'were not found in the provided directory ({deployments_path}). ' + 
                'Did you provide the right path via deployments_path?')
            return 

    deployment_mode = f'{deployment}-{mode}'
    year = year_path(project, deployment)

    glider_path = os.path.join(deployments_path, project, year, deployment)
    if not os.path.isdir(glider_path):
        _log.error(f'glider_path ({glider_path}) does not exist')
        return
    
    # if write_imagery:
    #     if not os.path.isdir(imagery_path):
    #         _log.error('write_imagery is true, and thus imagery_path ' + 
    #                       f'({imagery_path}) must be a valid path')
    #         return

    cacdir = os.path.join(deployments_path, 'cache')
    binarydir = os.path.join(glider_path, 'data', 'binary', mode)
    deploymentyaml = os.path.join(glider_path, 'config', 
        f"{deployment_mode}.yml")
    engyaml = get_engyml_path()

    ncdir = os.path.join(glider_path, 'data', 'nc')

    tsdir = os.path.join(ncdir, 'timeseries')
    profiledir = os.path.join(ncdir, 'ngdac', mode)
    griddir = os.path.join(ncdir, 'gridded')

    # return cacdir, binarydir, deploymentyaml, tsdir, profiledir, griddir
    return {
        "cacdir": cacdir,
        "binarydir": binarydir,
        "deploymentyaml": deploymentyaml,
        "engyaml": engyaml,
        "tsdir": tsdir,
        "profiledir": profiledir,
        "griddir": griddir
    }
