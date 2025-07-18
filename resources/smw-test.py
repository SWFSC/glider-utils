# Sam's testing script for esdglider modules/functions

import logging

import xarray as xr

from esdglider import acoustics, config, gcp, glider

deployment_info = {
    "deployment": "amlr03-20231128",
    "project": "FREEBYRD",
    "mode": "delayed",
    "min_dt": "2023-11-28 20:46",
}

deployment_bucket = "amlr-gliders-deployments-dev"
imagery_bucket = "amlr-gliders-imagery-raw-dev"
acoustics_bucket = "amlr-gliders-acoustics-dev"

base_path = "/home/sam_woodman_noaa_gov"
deployments_path = f"{base_path}/{deployment_bucket}"
imagery_path = f"{base_path}/{imagery_bucket}"
acoustics_path = f"{base_path}/{acoustics_bucket}"
config_path = f"{base_path}/glider-lab/deployment-configs"


# def scrape_sfmc():
#     glider.scrape_sfmc(
#         deployment=deployment,
#         project=project,
#         bucket=deployment_bucket,
#         # sfmc_path='/var/sfmc',
#         sfmc_path=f"{base_path}/sfmc",
#         gcpproject_id="ggn-nmfs-usamlr-dev-7b99",
#         secret_id="sfmc-swoodman",
#     )


# def prof(paths):
#     outname_tssci = os.path.join(paths["tsdir"], f"{deployment}-{mode}-sci.nc")
#     return glider.ngdac_profiles(
#         outname_tssci,
#         paths["profdir"],
#         paths["deploymentyaml"],
#         force=True,
#     )


def yaml():
    with open("db/glider-db-prod.txt", "r") as f:
        conn_string = f.read()
    return config.make_deployment_yaml(
        deployment_info,
        "C:/Users/sam.woodman/Downloads",
        conn_string,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(name)s:%(asctime)s:%(levelname)s:%(message)s [line %(lineno)d]",
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # logging.basicConfig(filename=logf,
    #                 filemode='w',
    #                 format='%(asctime)s %(levelname)-8s %(message)s',
    #                 level=logging.INFO,
    #                 datefmt='%Y-%m-%d %H:%M:%S')

    gcp.gcs_mount_bucket(deployment_bucket, deployments_path, ro=False)
    # gcp.gcs_mount_bucket(imagery_bucket, imagery_path, ro=False)
    gcp.gcs_mount_bucket(acoustics_bucket, acoustics_path, ro=False)
    paths = glider.get_path_deployment(
        deployment_info,
        deployments_path,
        config_path,
    )

    # ### Testing binary_to_raw
    # x = glider.binary_to_raw(
    #     paths["binarydir"],
    #     paths["cacdir"],
    #     paths["tsdir"],
    #     [paths["deploymentyaml"], paths["engyaml"]],
    #     search="*.[D|E|d|e][Bb][Dd]",
    #     fnamesuffix=f"-{mode}-raw",
    #     pp={
    #         "mode": mode,
    #         "min_dt": "2017-01-01",
    #         "file_info": None,
    #         "metadata_dict": {},
    #         "device_dict": {},
    #     },
    # )

    # scrape_sfmc()
    # yaml()

    ### Testing binary_to_nc
    outname = glider.binary_to_nc(
        deployment_info=deployment_info,
        paths=paths,
        write_raw=False,
        write_timeseries=False,
        write_gridded=False,
    )
    # prof(paths)

    dssci = xr.load_dataset(outname["outname_tssci"])
    # dseng = xr.load_dataset(outname_tseng)
    # dssci_g = xr.load_dataset(outname_5m)

    # ### Testing imagery
    # imagery.imagery_timeseries(
    #     dssci,
    #     imagery.get_path_imagery(project, deployment, imagery_path)
    # )

    ### Testing acoustics
    acoustics.echoview_metadata(
        dssci,
        acoustics.get_path_acoutics(deployment_info, acoustics_path),
    )

    # ### Testing plots
    # plots.all_loops(
    #     dssci, dseng, dssci_g, paths['plotdir'],
    #     os.path.join("/home/sam_woodman_noaa_gov", "ETOPO_2022_v1_15s_N45W135_erddap.nc")
    # )

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

    # plots.sci_surface_map(dssci_g, "temperature").show()
