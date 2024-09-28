#!/bin/bash

DEPLOYMENTS_PATH=/home/sam_woodman_noaa_gov/amlr-gliders-deployments-dev
IMAGERY_PATH=/home/sam_woodman_noaa_gov/amlr-gliders-imagery-raw-dev

fusermount -u $DEPLOYMENTS_PATH
gcsfuse --implicit-dirs amlr-gliders-deployments-dev $DEPLOYMENTS_PATH
echo -e "\nend of bucket mounting\n"

# conda activate esdglider
/opt/conda/envs/esdglider/bin/python glider-utils/scripts/binary_to_nc.py amlr08-20220513 SANDIEGO delayed $PATH_DEPLOYMENTS --loglevel=info --write_timeseries --write_gridded
