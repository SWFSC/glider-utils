# Changelog

All notable changes to the edgldier package will be documented in this file. See the [ESD glider lab manual](https://swfsc.github.io/glider-lab-manual/content/glider-data.html) for descriptions for processing funtionality, data products, etc.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Changed gridding wrapper function to ignore certain variables (note: currently requires https://github.com/smwoodman/pyglider)
- Changed `glider.binary_to_nc` to pass 'sci_water_pressure' to `pyglider.slocum.binary_to_timeseries`, rather than 'sci_water_temp'
- Changed `glider.binary_to_nc` so that it uses `pyglider.slocum.binary_to_timeseries` to generate the engineering timeseries for all deployments
- Added a new package data file 'deployment-raw-vars.yml'. This file contains contents of the 'deployment-eng-vars.yml' that should not be interpolated (e.g., commanded parameters). Changed `glider.binary_to_nc` to use this new file, and generalized `get_path_engyaml` to `get_path_yaml(yaml_type)` to get the path for either the eng or raw yaml.
- Fixed 'processing_level' attribute to be accurate across raw, engineering, and science timeseries
- Removed waypoint_latitude and waypoint_longitude from the default glider config files. As commanded variables, they will remain in the 'deployment-raw-vars.yml' file, and thus as part of the raw dataset
- Changed names TODO
- Changed `timeseries_raw_to_sci` to interpolate using nanoseconds, rather than rounding to seconds and then interpolating
- Changed the name of the measured depth in the raw data file to 'depth_measured'. This is consistent with 'depth_ctd', and will hopefully minimize confusion if 'depth_measured' is included in the science timeseries
- Changed `utils.calc_profile_summary` so that the user must 
- Changed `utils.check_profiles` so that it now takes a DataFrame (the output of `utils.calc_profile_summary`) as it's input, rather than the Dataset
- Changed tvt plots to use the raw dataset, so that all relevant sensors are present with new system


## [0.1.0] - 2025-07-17

- Initial release of esdglider, for basic processing of ESD glider data
