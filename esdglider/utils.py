import numpy as np
import logging
import collections
import datetime as dt

_log = logging.getLogger(__name__)


"""
ESD-specific utilities 
Mostly helpers for post-processing time series files created using pyglider
"""

def findProfiles(stamp: np.ndarray,depth: np.ndarray,**kwargs):
	"""
    Function copied exactly from:
    https://github.com/OceanGNS/PGPT/blob/main/scripts/gliderfuncs.py#L196

	Identify individual profiles and compute vertical direction from depth sequence.
	
	Args:
		stamp (np.ndarray): A 1D array of timestamps.
		depth (np.ndarray): A 1D array of depths.
		**kwargs (optional): Optional arguments including:
			- length (int): Minimum length of a profile (default=0).
			- period (float): Minimum duration of a profile (default=0).
			- inversion (float): Maximum depth inversion between cast segments of a profile (default=0).
			- interrupt (float): Maximum time separation between cast segments of a profile (default=0).
			- stall (float): Maximum range of a stalled segment (default=0).
			- shake (float): Maximum duration of a shake segment (default=0).
	
	Returns:
		profile_index (np.ndarray): A 1D array of profile indices.
		profile_direction (np.ndarray): A 1D array of vertical directions.
	"""
	if not (isinstance(stamp, np.ndarray) and isinstance(depth, np.ndarray)):
		stamp = stamp.to_numpy()
		depth = depth.to_numpy()
	
	# Flatten input arrays
	depth, stamp = depth.flatten(), stamp.flatten()
	
	# Check if the stamp is a datetime object and convert to elapsed seconds if necessary
	if np.issubdtype(stamp.dtype, np.datetime64):
		stamp = (stamp - stamp[0]).astype('timedelta64[s]').astype(float)
	
	# Set default parameter values (did not set type np.timedelta64(0, 'ns') )
	optionsList = { "length": 0, "period": 0, "inversion": 0, "interrupt": 0, "stall": 0, "shake": 0}
	optionsList.update(kwargs)
	
	validIndex = np.argwhere(np.logical_not(np.isnan(depth)) & np.logical_not(np.isnan(stamp))).flatten()
	validIndex = validIndex.astype(int)
	
	sdy = np.sign(np.diff(depth[validIndex], n=1, axis=0))
	depthPeak = np.ones(np.size(validIndex), dtype=bool)
	depthPeak[1:len(depthPeak) - 1,] = np.diff(sdy, n=1, axis=0) != 0
	depthPeakIndex = validIndex[depthPeak]
	sgmtFrst = stamp[depthPeakIndex[0:len(depthPeakIndex) - 1,]]
	sgmtLast = stamp[depthPeakIndex[1:,]]
	sgmtStrt = depth[depthPeakIndex[0:len(depthPeakIndex) - 1,]]
	sgmtFnsh = depth[depthPeakIndex[1:,]]
	sgmtSinc = sgmtLast - sgmtFrst
	sgmtVinc = sgmtFnsh - sgmtStrt
	sgmtVdir = np.sign(sgmtVinc)

	castSgmtValid = np.logical_not(np.logical_or(np.abs(sgmtVinc) <= optionsList["stall"], sgmtSinc <= optionsList["shake"]))
	castSgmtIndex = np.argwhere(castSgmtValid).flatten()
	castSgmtLapse = sgmtFrst[castSgmtIndex[1:]] - sgmtLast[castSgmtIndex[0:len(castSgmtIndex) - 1]]
	castSgmtSpace = -np.abs(sgmtVdir[castSgmtIndex[0:len(castSgmtIndex) - 1]] * (sgmtStrt[castSgmtIndex[1:]] - sgmtFnsh[castSgmtIndex[0:len(castSgmtIndex) - 1]]))
	castSgmtDirch = np.diff(sgmtVdir[castSgmtIndex], n=1, axis=0)
	castSgmtBound = np.logical_not((castSgmtDirch[:,] == 0) & (castSgmtLapse[:,] <= optionsList["interrupt"]) & (castSgmtSpace <= optionsList["inversion"]))
	castSgmtHeadValid = np.ones(np.size(castSgmtIndex), dtype=bool)
	castSgmtTailValid = np.ones(np.size(castSgmtIndex), dtype=bool)
	castSgmtHeadValid[1:,] = castSgmtBound
	castSgmtTailValid[0:len(castSgmtTailValid) - 1,] = castSgmtBound

	castHeadIndex = depthPeakIndex[castSgmtIndex[castSgmtHeadValid]]
	castTailIndex = depthPeakIndex[castSgmtIndex[castSgmtTailValid] + 1]
	castLength = np.abs(depth[castTailIndex] - depth[castHeadIndex])
	castPeriod = stamp[castTailIndex] - stamp[castHeadIndex]
	castValid = np.logical_not(np.logical_or(castLength <= optionsList["length"], castPeriod <= optionsList["period"]))
	castHead = np.zeros(np.size(depth))
	castTail = np.zeros(np.size(depth))
	castHead[castHeadIndex[castValid] + 1] = 0.5
	castTail[castTailIndex[castValid]] = 0.5

	profileIndex = 0.5 + np.cumsum(castHead + castTail)
	profileDirection = np.empty((len(depth,)))
	profileDirection[:] = np.nan

	for i in range(len(validIndex) - 1):
		iStart = validIndex[i]
		iEnd = validIndex[i + 1]
		profileDirection[iStart:iEnd] = sdy[i]

	return profileIndex, profileDirection



def get_fill_profiles(ds, time_vals, depth_vals):
    """
    Calculate profile index and direction values, 
    and fill values and attributes into ds

    ds : `xarray.Dataset`
    time_vals, depth_vals: passed directly to utils.findProfiles
    
    returns Dataset
    """

    prof_idx, prof_dir = findProfiles(
        time_vals, depth_vals, stall=20, shake=200)

    attrs = collections.OrderedDict([
        ('long_name', 'profile index'),
        ('units', '1'),
        ('comment',
         'N = inside profile N, N + 0.5 = between profiles N and N + 1'),
        ('sources', f'time depth'),
        ('method', 'esdglider.utils.findProfiles'),
        ('stall', 20),
        ('shake', 200)])
    ds['profile_index'] = (('time'), prof_idx, attrs)

    attrs = collections.OrderedDict([
        ('long_name', 'glider vertical speed direction'),
        ('units', '1'),
        ('comment',
         '-1 = ascending, 0 = inflecting or stalled, 1 = descending'),
        ('sources', f'time depth'),
        ('method', 'esdglider.utils.findProfiles')])
    ds['profile_direction'] = (('time'), prof_dir, attrs)
    
    # ds = utils.get_profiles_esd(ds, "depth")
    _log.debug(f"There are {np.max(ds.profile_index.values)} profiles")

    return ds


def drop_bogus(ds, ds_type, min_dt='2017-01-01'):
    """
    Remove/drop bogus times and values 

    Times from before min_dt are dropped; default is beofre Jan 2017, 
    which is beofre the ESD/AERD glider program. 

    ds: `xarray.Dataset`
        Dataset, with 'time' coordinate
    ds_type: string
        String of either 'sci' or 'eng'
    min_dt: string
        String to be passed to np.datetime64. Minimum datetime to keep.
        For instance, '1971-01-01', or '2020-03-06 12:00:00'
    
    Returns: filtered Dataset
    """

    if not (ds_type in ['sci', 'eng']):
        raise ValueError('ds_type must be either sci or eng')

    # For out of range or nan time/lat/lon, drop rows
    num_orig = len(ds.time)
    ds = ds.where(ds.time >= np.datetime64(min_dt), drop=True)
    if (num_orig-len(ds.time)) > 0:
        _log.info(f"Dropped {num_orig - len(ds.time)} times " +
                  f"that were either nan or before {min_dt}")

    num_orig = len(ds.time)
    ll_good = (
        (ds.longitude >= -180) & (ds.longitude <= 180) 
        & (ds.latitude >= -90) & (ds.latitude <= 90))
    ds = ds.where(ll_good, drop=True)
    if (num_orig-len(ds.time)) > 0:
        _log.info(f"Dropped {num_orig - len(ds.time)} nan " + 
                  "or out of range lat/lons")
    
    # For science variables, change out of range values to nan
    if ds_type == "sci":
        drop_values = {
            'conductivity':[0, 60], 
            'temperature':[-5, 100], 
            'pressure':[-2, 1500], 
            'chlorophyll':[0, 30], 
            'cdom':[0, 30], 
            'backscatter_700':[0, 5],  
            # 'oxygen_concentration':[-100, 500],
            'salinity':[0, 50], 
            'potential_density':[900, 1050], 
            'density':[1000, 1050], 
            'potential_temperature':[-5, 100]
        } 
        for var, value in drop_values.items():
            if not var in list(ds.keys()):
                _log.debug(f"{var} not present in ds - skipping drop_values check")
                continue
            num_orig = len(ds[var])
            good = (ds[var] >= value[0]) & (ds[var] <= value[1])
            ds[var] = ds[var].where(good, drop=False)
            if num_orig - len(ds[var]) > 0:
                _log.info(f"Changed {num_orig - len(ds[var])} {var} values " +
                        f"outside range [{value[0]}, {value[1]}] to nan")

            # num_orig = len(ds[var])
            # ds = ds.where(ds[var] <= value[1], drop=False)
            # _log.info(f"Changed {num_orig - len(ds[var])} {var} values greater than {value[1]} to nan")

    return ds


def solocam_filename_dt(filename, index_dt, format='%Y%m%d-%H%M%S'):
    """
    Parse imagery filename to return associated datetime
    Requires index of start of datetime part of string

    filename : str : Full filename
    index_start : int : The index of the start of the datetime string.
        The datetime runs from this index to this index plus 15 characters
    format : str : format passed to strptime

    Returns:
        Datetime object, with the datetime extracted from the imagery filename
    """
    solocam_substr = filename[index_dt:(index_dt+15)]
    _log.debug(f"datetime substring: {solocam_substr}")
    solocam_dt = dt.datetime.strptime(solocam_substr, format)

    return solocam_dt



def esd_file_id(ds):
    """
    ESD's version of pyglider.utils.get_file_id.
    This version does not require a glider_serial
    Make a file id for a Dataset: Id = *glider_name* + "YYYYMMDDTHHMM"
    """

    _log.debug(ds.time)
    if not ds.time.dtype == 'datetime64[ns]':
        dt = ds.time.values[0].astype('timedelta64[s]') + np.datetime64('1970-01-01')
    else:
        dt = ds.time.values[0].astype('datetime64[s]')
    _log.debug(f'dt, {dt}')
    id = (
        ds.attrs['glider_name']
        # + ds.attrs['glider_serial']
        + '-'
        + dt.item().strftime('%Y%m%dT%H%M')
    )
    return id


def data_var_reorder(ds, new_start):
    """
     Reorder the data variables of a dataset

     new_start is a list of the data variable names from ds that 
     will be moved to 'first' in the dataset

     Returns ds, with reordered data variables
    """

    ds_vars_orig = list(ds.data_vars)
    if not all([i in ds_vars_orig for i in new_start]):
        _log.error(f"new_start: {new_start}")
        _log.error(f"ds.data_vars: {ds_vars_orig}")
        raise ValueError("All values of new_start must be in ds.data_vars")
    
    new_order = new_start + [i for i in ds.data_vars if i not in new_start]
    ds = ds[new_order]
    
    # Double check that all values are present in new ds
    if not (all([j in ds_vars_orig for j in new_order] + 
                [j in new_order for j in ds_vars_orig])):
        raise ValueError("Error reordering data variables")

    return ds
