import logging
import os

from dbdreader.decompress import decompress_file, is_compressed

from esdglider import gcp, glider

"""
This script is intended to help users quickly generate decompressed
binary files, as a light wrapper around dbdreader functions.
"""

deployment_info = {
    "deployment": "unit_1024-20250224",
    "project": "SANDIEGO",
    "mode": "delayed",
    "min_dt": "2025-02-24",
}

if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s:%(asctime)s:%(levelname)s:%(message)s [line %(lineno)d]",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    bucket_name = "amlr-gliders-deployments-dev"
    deployments_path = os.path.join("/home/sam_woodman_noaa_gov", bucket_name)
    gcp.gcs_mount_bucket(bucket_name, deployments_path, ro=False)

    paths = glider.get_path_deployment(deployment_info, deployments_path, "")

    binarydir = paths["binarydir"]
    binarydir_files = os.listdir(binarydir)
    logging.info("There are %s files in %s", len(binarydir_files), binarydir)

    # FileDecompressor.decompress(dcd1)
    logging.info("decompressing all files in %s", binarydir)
    for fin in binarydir_files:
        logging.debug(fin)
        if is_compressed(fin):
            decompress_file(os.path.join(binarydir, fin))
        else:
            logging.debug("skipping %s", fin)

    binarydir_files = os.listdir(binarydir)
    logging.info("There are now %s files in %s", len(binarydir_files), binarydir)
