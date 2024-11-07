#!/bin/bash

# # In glider-proc VM:
# sudo touch /opt/glider-scripts/calanus-20241019-scrape_sfmc.sh
# sudo chmod 775 /opt/glider-scripts/calanus-20241019-scrape_sfmc.sh 
# sudo chgrp gliders /opt/glider-scripts/calanus-20241019-scrape_sfmc.sh

# Set variables
DEPLOYMENT=calanus-20241019
PROJECT=ECOSWIM

# Run scrape_sfmc script, using esdglider conda env
/opt/conda-envs/esdglider/bin/python /opt/glider-utils/scripts/scrape-sfmc.py $DEPLOYMENT $PROJECT --loglevel=info
