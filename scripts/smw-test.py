# Sam's testing script for esdglider modules/functions

import os
import logging
import xarray as xr

import esdglider.gcp as gcp
import esdglider.metadata as met
import esdglider.process as process


def scrape_sfmc():
    process.scrape_sfmc(deployment='calanus-20241019', 
        project="ECOSWIM", 
        bucket='amlr-gliders-deployments-dev', 
        # sfmc_path='/var/sfmc', 
        sfmc_path='/home/sam_woodman_noaa_gov/sfmc', 
        gcpproject_id='ggn-nmfs-usamlr-dev-7b99', 
        secret_id='sfmc-swoodman')


def ts():
    deployment = 'calanus-20241019'
    project = "ECOSWIM"
    mode = 'delayed'
    bucket_name = 'amlr-gliders-deployments-dev'

    deployments_path = f'/home/sam_woodman_noaa_gov/{bucket_name}'
    gcp.gcs_mount_bucket("amlr-gliders-deployments-dev", deployments_path, 
                         ro=False)

    x = process.binary_to_nc(
        deployment, project, mode, deployments_path, 
        write_timeseries=True, write_gridded=True, 
        min_dt='2024-10-19 17:00:00')
    
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

def yaml():
    conn_path = os.path.join(
        "C:/SMW/Gliders_Moorings/Gliders/glider-utils", 
        "db", "glider-db-prod.txt"
    )
    
    with open(conn_path, "r") as f:
        conn_string = f.read()

    return met.make_deployment_yaml(
        "amlr01-20181216", "FREEBYRD", "delayed", 
        "C:/Users/sam.woodman/Downloads", conn_string)


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

    
    # scrape_sfmc()
    # print(ts())
    # outname_tseng, outname_tssci, outname_1m, outname_5m = ts()
    print(yaml())

    # ds_eng = xr.load_dataset(outname_tseng)
    # ds_sci = xr.load_dataset(outname_tssci)
    # print(f"there are {len(ds_eng.time)} points in the engineering timeseries")
    # print(f"there are {len(ds_sci.time)} points in the science timeseries")
    # gcp.gcs_unmount_bucket("amlr-gliders-deployments-dev")
