#!/bin/bash

PATH_DEPLOYMENTS=/home/sam_woodman_noaa_gov/amlr-gliders-deployments-dev

fusermount -u $PATH_DEPLOYMENTS
gcsfuse --implicit-dirs amlr-gliders-deployments-dev $PATH_DEPLOYMENTS
echo -e "\nend of bucket mounting\n"

# conda activate esdglider
/opt/conda/envs/esdglider/bin/python glider-utils/scripts/binary_to_nc.py john-20240312 REFOCUS delayed $PATH_DEPLOYMENTS --loglevel=info
