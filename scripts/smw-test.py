# Sam's testing script for esdglider modules/functions

import os
import logging
import xarray as xr

import esdglider.utils as utils
import esdglider.metadata as met



def main():
    logging.basicConfig(
        format='%(module)s:%(levelname)s:%(message)s [line %(lineno)d]', 
        level=logging.INFO)
    
    # logging.basicConfig(filename=logf,
    #                 filemode='w',
    #                 format='%(asctime)s %(levelname)-8s %(message)s',
    #                 level=logging.INFO,
    #                 datefmt='%Y-%m-%d %H:%M:%S')


    deployment = 'amlr08-20220513'
    mode = 'delayed'
    bucket_name = 'amlr-gliders-deployments-dev'

    deployments_path = f'/home/sam_woodman_noaa_gov/{bucket_name}'
    utils.gcs_mount_bucket("amlr-gliders-deployments-dev", deployments_path, ro=True)
    glider_path = os.path.join(deployments_path, 'SANDIEGO', '2022', deployment)
    tsdir = os.path.join(glider_path, 'data', 'nc', 'timeseries')
    outname_dseng = os.path.join(tsdir, os.listdir(tsdir)[0])
    outname_dssci = os.path.join(tsdir, os.listdir(tsdir)[1])

    img_path = "/home/sam_woodman_noaa_gov/amlr-gliders-imagery-raw-dev"
    utils.gcs_mount_bucket("amlr-gliders-imagery-raw-dev", img_path, ro=True)
    i_path = os.path.join(img_path, "SANDIEGO", "2022", "amlr08-20220513", "images")
    i_fn = os.listdir(os.path.join(i_path, "Dir0002"))
    print(i_fn[0:5])


    #-------------------------
    for i in range(0, 5):
        print(met.solocam_filename_dt(i_fn[i], 5))

    dseng = xr.open_dataset(outname_dseng)
    dssci = xr.open_dataset(outname_dssci)
    met.imagery_metadata(dseng, dssci, i_path)

if __name__ == "__main__":
    main()
