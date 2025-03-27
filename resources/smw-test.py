# Sam's testing script for esdglider modules/functions

import os
import logging
import xarray as xr
# import esdglider
# import esdglider as eg

# # import esdglider.pathutils as putils
import esdglider.gcp as gcp
import esdglider.config as config
import esdglider.slocum as slocum
import esdglider.imagery as imagery
import esdglider.acoustics as acoustics
import esdglider.plots as plots


# deployment = 'calanus-20241019'
# project = "ECOSWIM"
# mode = 'delayed'

deployment = 'amlr08-20220513'
project = "SANDIEGO"
mode = 'delayed'

deployment_bucket = 'amlr-gliders-deployments-dev'
imagery_bucket    = 'amlr-gliders-imagery-raw-dev'
acoustics_bucket  = 'amlr-gliders-acoustics-dev'

base_path = "/home/sam_woodman_noaa_gov"
deployments_path = f'{base_path}/{deployment_bucket}'
imagery_path     = f'{base_path}/{imagery_bucket}'
acoustics_path   = f'{base_path}/{acoustics_bucket}'
config_path      = f"{base_path}/glider-lab/deployment-configs"


def scrape_sfmc():
    slocum.scrape_sfmc(deployment=deployment, 
        project=project, 
        bucket=deployment_bucket, 
        # sfmc_path='/var/sfmc', 
        sfmc_path=f'{base_path}/sfmc', 
        gcpproject_id='ggn-nmfs-usamlr-dev-7b99', 
        secret_id='sfmc-swoodman')


def ts(paths):
    x = slocum.binary_to_nc(
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
    return slocum.ngdac_profiles(
        outname_tssci, paths['profdir'], paths['deploymentyaml'], 
        force=True)

def yaml():
    with open("db/glider-db-prod.txt", "r") as f:
        conn_string = f.read()
    return config.make_deployment_config(
        deployment, project, mode, 
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

    gcp.gcs_mount_bucket(deployment_bucket, deployments_path, ro=False)
    gcp.gcs_mount_bucket(imagery_bucket, imagery_path, ro=False)
    gcp.gcs_mount_bucket(acoustics_bucket, acoustics_path, ro=False)
    paths = slocum.get_path_deployment(
        project, deployment, mode, deployments_path, config_path)
    
    # scrape_sfmc()
    # yaml()
    # outname_tseng, outname_tssci, outname_1m, outname_5m = ts(paths)
    # prof(paths)

    outname_tssci = os.path.join(paths['tsdir'], f"{deployment}-{mode}-sci.nc")
    dssci = xr.load_dataset(outname_tssci)
    # # Imagery    
    imagery.imagery_timeseries(
        dssci, 
        imagery.get_path_imagery(project, deployment, imagery_path)
    )

    # Acoustics
    acoustics.echoview_metadata(
        dssci, 
        acoustics.get_path_acoutics(project, deployment, acoustics_path)
    )

    # # Plots
    # outname_tseng = os.path.join(paths['tsdir'], f"{deployment}-{mode}-eng.nc")
    # dseng = xr.load_dataset(outname_tseng)

    # outname_grsci = os.path.join(paths['griddir'], f"{deployment}_grid-{mode}-5m.nc")
    # dssci_g = xr.load_dataset(outname_grsci)

    # base_path = paths['plotdir']
    # plots.sci_gridded_loop(dssci_g, base_path)
    # plots.sci_timeseries_loop(dssci, base_path)
    # plots.eng_timeseries_loop(dseng, base_path)
    # plots.eng_tvt_loop(dseng, base_path)
    # plots.sci_ts_loop(dssci, base_path)

    # bar_path = os.path.join("/home/sam_woodman_noaa_gov", "ETOPO_2022_v1_15s_N45W135_erddap.nc")
    # bar = xr.load_dataset(bar_path).rename({'latitude': 'lat', 'longitude': 'lon'})
    # bar = bar.where(bar.z <= 0, drop=True)
    # plots.sci_surface_map_loop(dssci_g, bar, base_path)
