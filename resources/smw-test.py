# Sam's testing script for esdglider modules/functions

import os
import logging
import xarray as xr

import esdglider as eg

# import esdglider.pathutils as putils
# import esdglider.gcp as gcp
# import esdglider.config as config
# import esdglider.slocum as process
# import esdglider.imagery as im


# deployment = 'calanus-20241019'
# project = "ECOSWIM"
# mode = 'delayed'

deployment = 'amlr08-20220513'
project = "SANDIEGO"
mode = 'delayed'

deployment_bucket = 'amlr-gliders-deployments-dev'
imagery_bucket = 'amlr-gliders-imagery-raw-dev'

base_path = "/home/sam_woodman_noaa_gov"
deployments_path = f'{base_path}/{deployment_bucket}'
imagery_path = f'{base_path}/{imagery_bucket}'
config_path = f"{base_path}/glider-lab/deployment-configs"



def scrape_sfmc():
    eg.slocum.scrape_sfmc(deployment=deployment, 
        project=project, 
        bucket=deployment_bucket, 
        # sfmc_path='/var/sfmc', 
        sfmc_path=f'{base_path}/sfmc', 
        gcpproject_id='ggn-nmfs-usamlr-dev-7b99', 
        secret_id='sfmc-swoodman')


def ts(paths):
    x = eg.slocum.binary_to_nc(
        deployment, mode, paths, # min_dt='2024-10-19 17:00:00'
        write_timeseries=True, write_gridded=True)
    
    return x

    # glider_path = os.path.join(deployments_path, 'SANDIEGO', '2022', deployment)
    # tsdir = os.path.join(glider_path, 'data', 'nc', 'timeseries')
    # outname_dseng = os.path.join(tsdir, os.listdir(tsdir)[0])
    # outname_dssci = os.path.join(tsdir, os.listdir(tsdir)[1])

    # img_path = "/home/sam_woodman_noaa_gov/amlr-gliders-imagery-raw-dev"
    # utils.gcs_mount_bucket("amlr-gliders-imagery-raw-dev", img_path, ro=True)
    # i_path = os.path.join(img_path, "SANDIEGO", "2022", "amlr08-20220513", "images")
    # i_fn = os.listdir(os.path.join(i_path, "Dir0002"))
    # print(i_fn[0:5])


    # #-------------------------
    # for i in range(0, 5):
    #     print(met.solocam_filename_dt(i_fn[i], 5))

    # dseng = xr.open_dataset(outname_dseng)
    # dssci = xr.open_dataset(outname_dssci)
    # met.imagery_metadata(dseng, dssci, i_path)

def prof(paths):    
    outname_tssci = os.path.join(paths['tsdir'], f"{deployment}-{mode}-sci.nc")
    return eg.slocum.ngdac_profiles(
        outname_tssci, paths['profdir'], paths['deploymentyaml'], 
        force=True)

def yaml():
    with open("db/glider-db-prod.txt", "r") as f:
        conn_string = f.read()
    return eg.config.make_deployment_config(
        deployment, project, mode, 
        "C:/Users/sam.woodman/Downloads", conn_string)

def imagery(ds_sci):
    eg.imagery.imagery_timeseries(
        ds_sci, 
        eg.imagery.imagery_paths(project, deployment, imagery_path)
    )


if __name__ == "__main__":
    logging.basicConfig(
        format='%(module)s:%(asctime)s:%(levelname)s:%(message)s [line %(lineno)d]', 
        level=logging.INFO, 
        datefmt='%Y-%m-%d %H:%M:%S')
    # logging.basicConfig(filename=logf,
    #                 filemode='w',
    #                 format='%(asctime)s %(levelname)-8s %(message)s',
    #                 level=logging.INFO,
    #                 datefmt='%Y-%m-%d %H:%M:%S')

    eg.gcp.gcs_mount_bucket(
        deployment_bucket, deployments_path, ro=False)
    eg.gcp.gcs_mount_bucket(
        imagery_bucket, imagery_path, ro=False)    
    paths = eg.slocum.get_path_esd(
        project, deployment, mode, deployments_path, config_path)
    
    # scrape_sfmc()
    # yaml()
    # outname_tseng, outname_tssci, outname_1m, outname_5m = ts(paths)
    # prof(paths)

    # Imagery
    outname_tssci = os.path.join(paths['tsdir'], f"{deployment}-{mode}-sci.nc")
    dssci = xr.open_dataset(outname_tssci)
    imagery(dssci)
