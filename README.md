# glider-utils

Utility functions for ESD glider processing.

This repo was inspired by [cproofutils](https://github.com/c-proof/cproofutils) and [votoutils](https://github.com/voto-ocean-knowledge/votoutils), as well as informed by experiences developing [amlr-gliders](https://github.com/us-amlr/amlr-gliders).

For more detailed information about the Ecosystem Science Division's (ESD) glider data processing, see the ESD glider lab manual: https://swfsc.github.io/glider-lab-manual/

## esdglider

This glider-utils repo contains the esdglider Python toolbox, which contains functionality for processing glider data from the ESD. To install and use this package, the recommended process is to create the esdglider conda environment, and then install the esdglider toolbox as editable. From the directory above where this repo is cloned:

```bash
# Create and activate the esdglider conda environment
conda env create -f glider-utils/environment.yml 
conda activate esdglider

# Install esdglider package
pip install -e glider-utils
```

You can then use esdglider functions in your scripts. For instance:

```python
from esdglider.process import binary_to_nc
```

### Modules

* **data**: folder for data included in the package
* **gcp**: functions specific to interacting with ESD's Google Cloud Platform project
* **metadata**: functions for creating ESD metadata files
* **pathutils**: utility functions related to directory and file path creation and management. These functions follow the directory structure outlined [here](https://swfsc.github.io/glider-lab-manual/content/data-management.html)
* **process**: processing functions, for instance scraping data from the SFMC or writing NetCDF files
* **utils**: utility functions for glider data shenanigans

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
