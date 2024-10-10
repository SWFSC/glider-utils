import collections
import numpy as np
import logging

from scipy.signal import argrelextrema

_log = logging.getLogger(__name__)


"""
ESD-specific utilities 
Mostly for post-processing time series files created using pyglider
"""


def get_profiles_esd(ds, depth_var='pressure', 
                     min_dp=10.0, filt_time=100, profile_min_time=300):
    """
    NOTE: this function is from pyglider.utils.get_profiles_new. 
    ESD needed a function that calculated profiles using user-defined depth var

    Find profiles in a glider timeseries:

    Parameters
    ----------
    ds : `xarray.Dataset`
        Must have *time* coordinate and depth_var as variables
    min_dp : float, default=10.0
        Minimum distance a profile must transit to be considered a profile, in dbar.
    filt_time : float, default=100
        Approximate length of time filter, in seconds.  Note that the filter
        is really implemented by sample, so the number of samples is
        ``filt_time / dt``
        where *dt* is the median time between samples in the time series.
    profile_min_time : float, default=300
        Minimum time length of profile in s.
    """

    x = ds[depth_var]

    profile = x.values * 0
    direction = x.values * 0
    pronum = 1

    good = np.where(np.isfinite(x))[0]
    dt = float(np.median(
        np.diff(ds.time.values[good[:200000]]).astype(np.float64)) * 1e-9)
    _log.info(f'dt, {dt}')
    filt_length = int(filt_time / dt)

    min_nsamples = int(profile_min_time / dt)
    _log.info('Filt Len  %d, dt %f, min_n %d', filt_length, dt, min_nsamples)
    if filt_length > 1:
        p = np.convolve(x.values[good],
                        np.ones(filt_length) / filt_length, 'same')
    else:
        p = x.values[good]
    decim = int(filt_length / 3)
    if decim < 2:
        decim = 2
    # why?  because argrelextrema doesn't like repeated values, so smooth
    # then decimate to get fewer values:
    pp = p[::decim]
    maxs = argrelextrema(pp, np.greater)[0]
    mins = argrelextrema(pp, np.less)[0]
    mins = good[mins * decim]
    maxs = good[maxs * decim]
    if mins[0] > maxs[0]:
        mins = np.concatenate(([0], mins))
    if mins[-1] < maxs[-1]:
        mins = np.concatenate((mins, good[[-1]]))

    _log.debug(f'mins: {len(mins)} {mins} , maxs: {len(maxs)} {maxs}')

    pronum = 0
    p = x
    nmin = 0
    nmax = 0
    while (nmin < len(mins)) and (nmax < len(maxs)):
        nmax = np.where(maxs > mins[nmin])[0]
        if len(nmax) >= 1:
            nmax = nmax[0]
        else:
            break
        _log.debug(nmax)
        ins = range(int(mins[nmin]), int(maxs[nmax]+1))
        _log.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
        _log.debug(f'Down, {ins}, {p[ins[0]].values},{p[ins[-1]].values}')
        if ((len(ins) > min_nsamples) and
                (np.nanmax(p[ins]) - np.nanmin(p[ins]) > min_dp)):
            profile[ins] = pronum
            direction[ins] = +1
            pronum += 1
        nmin = np.where(mins > maxs[nmax])[0]
        if len(nmin) >= 1:
            nmin = nmin[0]
        else:
            break
        ins = range(maxs[nmax], mins[nmin])
        _log.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
        _log.debug(f'Up, {ins}, {p[ins[0]].values}, {p[ins[-1]].values}')
        if ((len(ins) > min_nsamples) and
                (np.nanmax(p[ins]) - np.nanmin(p[ins]) > min_dp)):
            # up
            profile[ins] = pronum
            direction[ins] = -1
            pronum += 1

    attrs = collections.OrderedDict([
        ('long_name', 'profile index'),
        ('units', '1'),
        ('comment',
         'N = inside profile N, N + 0.5 = between profiles N and N + 1'),
        ('sources', f'time {depth_var}'),
        ('method', 'get_profiles_esd'),
        ('min_dp', min_dp),
        ('filt_length', filt_length),
        ('min_nsamples', min_nsamples)])
    ds['profile_index'] = (('time'), profile, attrs)

    attrs = collections.OrderedDict([
        ('long_name', 'glider vertical speed direction'),
        ('units', '1'),
        ('comment',
         '-1 = ascending, 0 = inflecting or stalled, 1 = descending'),
        ('sources', f'time {depth_var}'),
        ('method', 'get_profiles_esd')])
    ds['profile_direction'] = (('time'), direction, attrs)
    return ds


def drop_bogus_times(ds, min_dt='1971-01-01'):
    """
    Remove/drop times from before a given value. 
    By default this drops 1970-01-01 timestamps, but can also be used to 
    drop uninformative timestamps from before when the glider was deployed

    
    ds : `xarray.Dataset`
        Dataset, with 'time' coordinate
    min_dt: string
        String to be passed to np.datetime64. Minimum dt to keep
    
    Returns: filtered Dataset
    """
    # ds = ds.sel(time=slice(min_dt, None))
    num_times_orig = len(ds.time)
    ds = ds.where(ds.time >= np.datetime64(min_dt), drop=True)
    _log.info(f"Dropped {num_times_orig - len(ds.time)} times from before {min_dt}")

    return ds
