import numpy as np
import logging
import collections
# from scipy.signal import argrelextrema

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


# def get_profiles_esd(ds, depth_var='pressure', 
#                      min_dp=5.0, filt_time=150, profile_min_time=300):
#     """
#     NOTE: this function is from pyglider.utils.get_profiles_new. 
#     ESD needed a function that calculated profiles 
#     using user-defined depth variable

#     Find profiles in a glider timeseries:

#     Parameters
#     ----------
#     ds : `xarray.Dataset`
#         Must have *time* coordinate and depth_var as variables
#     min_dp : float, default=10.0
#         Minimum distance a profile must transit to be considered a profile, in dbar.
#     filt_time : float, default=100
#         Approximate length of time filter, in seconds.  Note that the filter
#         is really implemented by sample, so the number of samples is
#         ``filt_time / dt``
#         where *dt* is the median time between samples in the time series.
#     profile_min_time : float, default=300
#         Minimum time length of profile in s.
#     """

#     x = ds[depth_var]

#     profile = x.values * 0
#     direction = x.values * 0
#     pronum = 1

#     good = np.where(np.isfinite(x))[0]
#     dt = float(np.median(
#         np.diff(ds.time.values[good[:200000]]).astype(np.float64)) * 1e-9)
#     _log.info(f'dt, {dt}')
#     filt_length = int(filt_time / dt)

#     min_nsamples = int(profile_min_time / dt)
#     _log.info('Filt Len  %d, dt %f, min_n %d', filt_length, dt, min_nsamples)
#     if filt_length > 1:
#         p = np.convolve(x.values[good],
#                         np.ones(filt_length) / filt_length, 'same')
#     else:
#         p = x.values[good]
#     decim = int(filt_length / 3)
#     if decim < 2:
#         decim = 2
#     # why?  because argrelextrema doesn't like repeated values, so smooth
#     # then decimate to get fewer values:
#     pp = p[::decim]
#     # pp = p
#     maxs = argrelextrema(pp, np.greater)[0]
#     mins = argrelextrema(pp, np.less)[0]
#     mins = good[mins * decim]
#     maxs = good[maxs * decim]
#     # mins = good[mins]
#     # maxs = good[maxs]
#     if mins[0] > maxs[0]:
#         mins = np.concatenate(([0], mins))
#     if mins[-1] < maxs[-1]:
#         mins = np.concatenate((mins, good[[-1]]))

#     _log.debug(f'mins: {len(mins)} {mins} , maxs: {len(maxs)} {maxs}')

#     # pronum = 0
#     p = x
#     nmin = 0
#     nmax = 0
#     while (nmin < len(mins)) and (nmax < len(maxs)):
#         nmax = np.where(maxs > mins[nmin])[0]
#         if len(nmax) >= 1:
#             nmax = nmax[0]
#         else:
#             break
#         _log.debug(nmax)
#         ins = range(int(mins[nmin]), int(maxs[nmax]+1))
#         _log.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
#         _log.debug(f'Down, {ins}, {p[ins[0]].values},{p[ins[-1]].values}')
#         if ((len(ins) > min_nsamples) and
#                 (np.nanmax(p[ins]) - np.nanmin(p[ins]) > min_dp)):
#             profile[ins] = pronum
#             direction[ins] = +1
#             pronum += 1
#         nmin = np.where(mins > maxs[nmax])[0]
#         if len(nmin) >= 1:
#             nmin = nmin[0]
#         else:
#             break
#         ins = range(maxs[nmax], mins[nmin])
#         _log.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
#         _log.debug(f'Up, {ins}, {p[ins[0]].values}, {p[ins[-1]].values}')
#         if ((len(ins) > min_nsamples) and
#                 (np.nanmax(p[ins]) - np.nanmin(p[ins]) > min_dp)):
#             # up
#             profile[ins] = pronum
#             direction[ins] = -1
#             pronum += 1

#     attrs = collections.OrderedDict([
#         ('long_name', 'profile index'),
#         ('units', '1'),
#         ('comment',
#          'N = insidesie profile N, N + 0.5 = between profiles N and N + 1'),
#         ('sources', f'time {depth_var}'),
#         ('method', 'get_profiles_esd'),
#         ('min_dp', min_dp),
#         ('filt_length', filt_length),
#         ('min_nsamples', min_nsamples)])
#     ds['profile_index'] = (('time'), profile, attrs)

#     attrs = collections.OrderedDict([
#         ('long_name', 'glider vertical speed direction'),
#         ('units', '1'),
#         ('comment',
#          '-1 = ascending, 0 = inflecting or stalled, 1 = descending'),
#         ('sources', f'time {depth_var}'),
#         ('method', 'get_profiles_esd')])
#     ds['profile_direction'] = (('time'), direction, attrs)
#     return ds


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

    # ds = ds.sel(time=slice(min_dt, None))
    num_times_orig = len(ds.time)
    ds = ds.where(ds.time >= np.datetime64(min_dt), drop=True)
    _log.info(f"Checking for times that are either nan or before {min_dt}")
    _log.info(f"Dropped {num_times_orig - len(ds.time)} times")

    # vars_to_check = ['conductivity', 'temperature', 'pressure', 'chlorophyll', 
    #                  'cdom', 'backscatter_700', 'salinity', 
    #                  'potential_density', 'density', 'potential_temperature'] 
    # 'oxygen_concentration',
    drop_values = {'conductivity':[0, 60], 
                   'temperature':[-5, 100], 
                   'pressure':[-2, 1500], 
                   'chlorophyll':[0, 30], 
                   'cdom':[0, 30], 
                   'backscatter_700':[0, 5],  
                   'salinity':[0, 50], 
                   'potential_density':[900, 1050], 
                   'density':[1000, 1050], 
                   'potential_temperature':[-5, 100]} 
    # 'oxygen_concentration':[-100, 500],
    
    if ds_type == "sci":
        for var, value in drop_values.items():
            if not var in list(ds.keys()):
                _log.debug(f"{var} not present in ds - skipping drop_values check")
                continue
            num_orig = len(ds[var])
            ds = ds.where((ds[var] >= value[0]) & (ds[var] <= value[1]), 
                          drop=False)
            _log.info(f"Changed {num_orig - len(ds[var])} {var} values " +
                      f"outside range [{value[0]}, {value[1]}] to nan")

            # num_orig = len(ds[var])
            # ds = ds.where(ds[var] <= value[1], drop=False)
            # _log.info(f"Changed {num_orig - len(ds[var])} {var} values greater than {value[1]} to nan")

    return ds
