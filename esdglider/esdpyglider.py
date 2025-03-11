# pyglider functions adapted for use by ESD scientists
# pyglider: https://github.com/c-proof/pyglider

import os
import logging
import numpy as np
import xarray as xr
import netCDF4

import pyglider.utils as utils

_log = logging.getLogger(__name__)

def esd_extract_timeseries_profiles(inname, outdir, deploymentyaml, force=False):
    """
    ESD's version of extract_timeseries_profiles, from:
    https://github.com/c-proof/pyglider/blob/main/pyglider/ncprocess.py#L19
    
     Extract and save each profile from a timeseries netCDF.

    Parameters
    ----------
    inname : str or Path
        netcdf file to break into profiles

    outdir : str or Path
        directory to place profiles

    deploymentyaml : str or Path
        location of deployment yaml file for the netCDF file.  This should
        be the same yaml file that was used to make the timeseries file.

    force : bool, default False
        Force an overwite even if profile netcdf already exists
    """
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    deployment = utils._get_deployment(deploymentyaml)

    meta = deployment['metadata']
    with xr.open_dataset(inname) as ds:
        _log.info('Extracting profiles: opening %s', inname)
        profiles = np.unique(ds.profile_index)
        profiles = [p for p in profiles if (~np.isnan(p) and not (p % 1) and (p > 0))]
        for p in profiles:
            ind = np.where(ds.profile_index == p)[0]
            dss = ds.isel(time=ind)
            outname = outdir + '/' + utils.get_file_id(dss) + '.nc'
            _log.info('Checking %s', outname)
            if force or (not os.path.exists(outname)):
                # this is the id for the whole file, not just this profile..
                dss['trajectory'] = utils.get_file_id(ds).encode()
                trajlen = len(utils.get_file_id(ds).encode())
                dss['trajectory'].attrs['cf_role'] = 'trajectory_id'
                dss['trajectory'].attrs['comment'] = (
                    'A trajectory is a single'
                    'deployment of a glider and may span multiple data files.'
                )
                dss['trajectory'].attrs['long_name'] = 'Trajectory/Deployment Name'

                # profile-averaged variables....
                profile_meta = deployment['profile_variables']
                if 'water_velocity_eastward' in dss.keys():
                    dss['u'] = dss.water_velocity_eastward.mean()
                    dss['u'].attrs = profile_meta['u']

                    dss['v'] = dss.water_velocity_northward.mean()
                    dss['v'].attrs = profile_meta['v']
                elif 'u' in profile_meta:
                    dss['u'] = profile_meta['u'].get('_FillValue', np.nan)
                    dss['u'].attrs = profile_meta['u']

                    dss['v'] = profile_meta['v'].get('_FillValue', np.nan)
                    dss['v'].attrs = profile_meta['v']
                else:
                    dss['u'] = np.nan
                    dss['v'] = np.nan

                dss['profile_id'] = np.int32(p)
                dss['profile_id'].attrs = profile_meta['profile_id']
                if '_FillValue' not in dss['profile_id'].attrs:
                    dss['profile_id'].attrs['_FillValue'] = -1
                dss['profile_id'].attrs['valid_min'] = np.int32(
                    dss['profile_id'].attrs['valid_min']
                )
                dss['profile_id'].attrs['valid_max'] = np.int32(
                    dss['profile_id'].attrs['valid_max']
                )

                dss['profile_time'] = dss.time.mean()
                dss['profile_time'].attrs = profile_meta['profile_time']
                # remove units so they can be encoded later:
                try:
                    del dss.profile_time.attrs['units']
                    del dss.profile_time.attrs['calendar']
                except KeyError:
                    pass
                dss['profile_lon'] = dss.longitude.mean()
                dss['profile_lon'].attrs = profile_meta['profile_lon']
                dss['profile_lat'] = dss.latitude.mean()
                dss['profile_lat'].attrs = profile_meta['profile_lat']

                dss['lat'] = dss['latitude']
                dss['lon'] = dss['longitude']
                dss['platform'] = np.int32(1)
                comment = meta['glider_model'] + ' operated by ' + meta['institution']
                dss['platform'].attrs['comment'] = comment
                dss['platform'].attrs['id'] = (
                    meta['glider_name'] + meta['glider_serial']
                )
                dss['platform'].attrs['instrument'] = 'instrument_ctd'
                dss['platform'].attrs['long_name'] = (
                    meta['glider_model'] + dss['platform'].attrs['id']
                )
                dss['platform'].attrs['type'] = 'platform'
                dss['platform'].attrs['wmo_id'] = meta['wmo_id']
                if '_FillValue' not in dss['platform'].attrs:
                    dss['platform'].attrs['_FillValue'] = -1

                dss['lat_uv'] = np.nan
                dss['lat_uv'].attrs = profile_meta['lat_uv']
                dss['lon_uv'] = np.nan
                dss['lon_uv'].attrs = profile_meta['lon_uv']
                dss['time_uv'] = np.nan
                dss['time_uv'].attrs = profile_meta['time_uv']

                dss['instrument_ctd'] = np.int32(1.0)
                dss['instrument_ctd'].attrs = profile_meta['instrument_ctd']
                if '_FillValue' not in dss['instrument_ctd'].attrs:
                    dss['instrument_ctd'].attrs['_FillValue'] = -1

                dss.attrs['date_modified'] = str(np.datetime64('now')) + 'Z'

                # ancillary variables: link and create with values of 2.  If
                # we dont' want them all 2, then create these variables in the
                # time series
                to_fill = [
                    'temperature',
                    'pressure',
                    'conductivity',
                    'salinity',
                    'density',
                    'lon',
                    'lat',
                    'depth',
                ]
                for name in to_fill:
                    qcname = name + '_qc'
                    dss[name].attrs['ancillary_variables'] = qcname
                    if qcname not in dss.keys():
                        dss[qcname] = ('time', 2 * np.ones(len(dss[name]), np.int8))
                        dss[qcname].attrs = utils.fill_required_qcattrs({}, name)
                        # 2 is "not eval"
                # outname = outdir + '/' + utils.get_file_id(dss) + '.nc'
                _log.info('Writing %s', outname)
                timeunits = 'seconds since 1970-01-01T00:00:00Z'
                timecalendar = 'gregorian'
                try:
                    del dss.profile_time.attrs['_FillValue']
                    del dss.profile_time.attrs['units']
                except KeyError:
                    pass
                dss.to_netcdf(
                    outname,
                    encoding={
                        'time': {
                            'units': timeunits,
                            'calendar': timecalendar,
                            'dtype': 'float64',
                        },
                        'profile_time': {
                            'units': timeunits,
                            '_FillValue': -99999.0,
                            'dtype': 'float64',
                        },
                    },
                )

                # add traj_strlen using bare ntcdf to make IOOS happy
                with netCDF4.Dataset(outname, 'r+') as nc:
                    nc.renameDimension('string%d' % trajlen, 'traj_strlen')



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
