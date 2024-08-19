#!/bin/bash

PATH_DEPLOYMENTS=/home/sam_woodman_noaa_gov/amlr-gliders-deployments-dev


# gcsfuse --implicit-dirs amlr-gliders-deployments-dev $PATH_DEPLOYMENTS

# conda activate esdgliderutils
/opt/conda/envs/esdgliderutils/bin/python glider-utils/scripts/binary_to_nc.py john-20240312 REFOCUS delayed $PATH_DEPLOYMENTS --loglevel=info
