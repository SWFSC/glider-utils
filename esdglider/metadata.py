import os
import logging
import datetime as dt
import glob
import pandas as pd

logger = logging.getLogger(__name__)


def solocam_filename_dt(filename, index_start):
    """
    Parse imagery filename to return associated datetime
    Requires index of start of datetime part of string-
    """
    solocam_substr = filename[index_start:(index_start+15)]
    solocam_dt = dt.datetime.strptime(solocam_substr, '%Y%m%d-%H%M%S')

    return solocam_dt


# def imagery_metadata(ds, deployment, glider_path, imagery_path, ext = 'jpg'):
#     """
#     Matches up imagery files with data from gdm object by imagery filename
#     Uses interpolated variables (hardcoded in function)
#     Returns data frame with metadata information
    
#     Args:
#         ds (Dataset): xarray Dataset; output of binary_to_timeseries
#         deployment (str): deployment name
#         glider_path (str): path to 'glider' folder within deployment folder
#         imagery_path (str): path to folder with images, 
#             specifically the 'Dir####' folders
#         ext (str, optional): Imagery file extension. Defaults to 'jpg'.

#     Returns:
#         DataFrame: DataFrame of imagery metadata
#     """
    
#     logger.info(f'Creating imagery metadata file for {deployment}')

#     lat_column = 'ilatitude'
#     lon_column = 'ilongitude'
#     depth_column = 'idepth'
#     pitch_column = 'impitch'
#     roll_column = 'imroll'


#     #--------------------------------------------
#     # Checks
#     out_path = os.path.join(glider_path, 'data', 'out', 'cameras')
#     if not os.path.exists(out_path):
#         logger.info(f'Creating directory at: {out_path}')
#         os.makedirs(out_path)

#     if not os.path.isdir(imagery_path):
#         logger.error(f'imagery_path ({imagery_path}) does not exist, and thus the ' + 
#                         'CSV file with imagery metadata will not be created')
#         return
#     else:
#         imagery_filepaths = glob.glob(f'{imagery_path}/**/*.{ext}', recursive=True)
#         imagery_files = [os.path.basename(x) for x in imagery_filepaths]
#         imagery_files.sort()

#         # TODO: check for non-sensical file paths\

#     #--------------------------------------------
#     logger.info("Creating timeseries for imagery processing")
#     imagery_vars_list = [lat_column, lon_column, 
#         depth_column, pitch_column, roll_column]
#     imagery_vars_set = set(imagery_vars_list)

#     if not imagery_vars_set.issubset(gdm.data.columns):
#         logger.error('gdm object does not contain all required columns. ' + 
#             f"Missing columns: {', '.join(imagery_vars_set.difference(gdm.data.columns))}")
#         return()

#     gdm.data = gdm.data[imagery_vars_list]
#     ds = gdm.data.to_xarray()


#     #--------------------------------------------
#     # Extract info from imagery file names, and match up with glider data
#     logger.info("Processing imagery file names")

#     # Check that all filenames have the same number of characters
#     if not len(set([len(i) for i in imagery_files])) == 1:
#         logger.warning('The imagery file names are not all the same length, ' + 
#             'and thus shuld be checked carefully')
#         # return()


#     space_index = str.index(imagery_files[0], ' ')
#     if space_index == -1:
#         logger.error('The imagery file name year index could not be found, ' + 
#             'and thus the imagery metadata file cannot be generated')
#         return()
#     yr_index = space_index + 1   

#     try:
#         imagery_file_dts = [solocam_filename_dt(i, yr_index) for i in imagery_files]
#     except:
#         logger.error(f'Datetimes could not be extracted from imagery filenames ' + 
#                         '(at {deployment_imagery_path}), and thus the ' + 
#                         'CSV file with imagery metadata will not be created')
#         return


#     imagery_dict = {'img_file': imagery_files, 'img_dt': imagery_file_dts}
#     imagery_df = pd.DataFrame(data = imagery_dict).sort_values('img_dt')

#     logger.info("Finding nearest glider data slice for each imagery datetime")
#     # ds_nona = ds.sel(time = ds.depth.dropna('time').time.values)
#     # TODO: check if any time values are NA
#     ds_slice = ds.sel(time=imagery_df.img_dt.values, method = 'nearest')

#     imagery_df['glider_dt'] = ds_slice.time.values
#     diff_dts = (imagery_df.img_dt - imagery_df.glider_dt).astype('timedelta64[s]')
#     imagery_df['diff_dt_seconds'] = diff_dts.dt.total_seconds()
    
#     imagery_df['latitude'] = ds_slice[lat_column].values
#     imagery_df['longitude'] = ds_slice[lon_column].values
#     imagery_df['depth'] = ds_slice[depth_column].values
#     imagery_df['pitch'] = ds_slice[pitch_column].values
#     imagery_df['roll'] = ds_slice[roll_column].values

#     # csv_file = os.path.join(out_path, f'{deployment}-imagery-metadata.csv')
#     # logger.info(f'Writing imagery metadata to: {csv_file}')
#     # imagery_df.to_csv(csv_file, index=False)
    
#     csv_file = os.path.join(imagery_path, f'{deployment}-imagery-metadata.csv')
#     logger.info(f'Writing imagery metadata to: {csv_file}')
#     imagery_df.to_csv(csv_file, index=False)

#     return imagery_df