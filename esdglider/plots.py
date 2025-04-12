import logging
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.gridspec import GridSpec

from esdglider import utils

_log = logging.getLogger(__name__)

"""Constants used throughout plotting functions
"""
label_size = 11
title_size = 13

"""Helper functions and dictionaries for plot formats/display

Parameters
----------
ds: xarray Dataset
    Requisite dataset
var: str
    Variable name as a string
"""


# def return_var(x):
#     """Simply return a variable"""
#     return x


def adj_var(ds, var):
    """Get the adjusted var values for the plot. Eg, take the log"""
    if var not in adjustments.keys():
        return ds[var]
    elif adjustments[var] == np.log10:
        return adjustments[var](ds[var])


def adj_var_label(ds, var):
    """Get the formatted log label for the plot"""
    u = ds[var].attrs["units"]
    # if var not in list(units.keys()):
    #     u2 = "missing"
    # else:
    #     u2 = units[var]
    # print(f"var: {var}; ds: {u}; dict: {u2}")

    # if var in adjustments_labels.keys():
    #     return f"{adjustments_labels[var]}({var} [{u}])"
    # else:
    #     return f"{var} [{u}]"
    
    if var not in adjustments.keys():
        return f"{var} [{u}]"
    elif adjustments[var] == np.log10:
        return "$log_{10}$"+f"({var} [{u}])"
    # elif adjustments[var] == return_var:
    #     return f"{var} [{u}]"
    else:
        raise ValueError(f"var {var}: not an expected adjustments value")


"""Dictionary of adjustments to make to variables to plot
"""
adjustments = {
    # "temperature": return_var,
    "chlorophyll": np.log10,
    "cdom": np.log10,
    "backscatter_700": np.log10,
    # "oxygen_concentration": return_var,
    # "salinity": return_var,
    # "density": return_var,
}

# """ Dictionary of label adjustments; correspnds to adjustments dict
# """
# adjustments_labels = {
#     "chlorophyll": "$log_{10}$",
#     "cdom": "$log_{10}$",
#     "backscatter_700": "$log_{10}$",
# }

# """ Dictionary of variable units
# SMW note: This should likely be changed to use units from variable attributes
# """
# units = {
#     "latitude": "deg",
#     "longitude": "deg",
#     "depth": "m",
#     "heading": "deg",
#     "pitch": "rad",
#     "roll": "rad",
#     "waypoint_latitude": "deg",
#     "waypoint_longitude": "deg",
#     "conductivity": "S$\\bullet m^{-1}$",
#     "temperature": "°C",
#     "pressure": "dbar",
#     "chlorophyll": "µg•$L^{-l}$",
#     "cdom": "ppb",
#     "backscatter_700": "$m^{-1}$",
#     "oxygen_concentration": "µmol•$L^{-1}$",
#     "depth_ctd": "m",
#     "distance_over_ground": "km",
#     "salinity": "PSU",
#     "potential_density": "kg $\\bullet m^{-3}$",
#     "density": "kg $\\bullet m^{-3}$",
#     "potential_temperature": "°C",
#     "profile_index": "1",
#     "profile_direction": "1",
#     "total_num_inflections": "1",
# }

"""Dictionary of the colors to use for different variables
"""
sci_colors = {
    "cdom": cmo.solar,
    "chlorophyll": cmo.algae,
    "oxygen_concentration": cmo.tempo,
    "backscatter_700": colormaps["terrain"],
    "temperature": cmo.thermal,
    "potential_temperature": cmo.thermal,
    "salinity": cmo.haline,
    "density": colormaps["cividis"],
    "potential_density": colormaps["cividis"],
    "profile_index": cmo.gray,
}

"""List of science variables
Plotting functions loop through this list to plot science vars
"""
sci_vars = [
    "temperature",
    "salinity",
    "density",
    "chlorophyll",
    "cdom",
    "oxygen_concentration",
    "backscatter_700",
]

"""List of engineering variables
Plotting functions loop through this list to plot engineering vars
"""
eng_vars = [
    "heading",
    "pitch",
    "roll",
    "total_num_inflections",
    "commanded_oil_volume",
    "measured_oil_volume",
    "total_amphr",
    "amphr",
    "battery_voltage",
    "vacuum",
    "leak_detect",
    "leak_detect_forward",
    "leak_detect_science",
    "battpos",
    "target_depth",
    "altitude",
    "distance_over_ground",
    "profile_index",
    "profile_direction",
]


def all_loops(
    dssci: xr.Dataset,
    dseng: xr.Dataset,
    dssci_g: xr.Dataset,
    crs = None,
    base_path: str = None,
    bar_file: str | None = None,
):
    """
    Wrapper to run all of the plotting loop functions

    Parameters
    ----------
    dssci : xarray Dataset
        Timeseries science dataset
    dseng : xarray Dataset
        Timeseries engineering dataset
    dssci_g : xarray Dataset
        Gridded science dataset
    crs : a class from cartopy.crs or None; default None
        An instantiated cartopy projection, such as cartopy.crs.PlateCarree()
        or cartopy.crs.Mercator(). 
        If None, surface maps are not created
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        Nothing
    """

    _log.info("Doing all of the loops")
    utils.rmtree(base_path)

    sci_gridded_loop(dssci_g, base_path)
    sci_timeseries_loop(dssci, base_path)
    eng_timeseries_loop(dseng, base_path)
    eng_tvt_loop(dseng, base_path)
    sci_ts_loop(dssci, base_path)

    if bar_file is not None:
        bar = xr.load_dataset(bar_file).rename({"latitude": "lat", "longitude": "lon"})
        bar = bar.where(bar.z <= 0, drop=True)
    else:
        _log.info("No bar file path, so skipping surface maps")
        bar = None

    if crs is not None:
        sci_surface_map_loop(dssci_g, crs=crs, base_path=base_path, bar=bar)
    else: 
        _log.info("No crs provided, and thus skipping surface maps")


def sci_gridded_loop(
    ds: xr.Dataset,
    base_path: str | None = None,
    show: bool = False,
):
    """
    A loop/wrapper function to use a gridded science dataset to make plots
    of all sci_vars variables.
    Specifically, for each sci_var present in the dataset,
    create timesection, spatialsection, and spatialgrid plots

    Arguments let the user specify if these plots should be saved, and/or shown

    Parameters
    ----------
    ds : xarray Dataset
        Gridded science dataset
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        Nothing
    """

    # plt.scatter(sci_ds_g.time, sci_ds_g.profile)
    _log.info("LOOP: making gridded science plots")
    for var in sci_vars:
        _log.debug(f"var {var}")
        if var not in list(ds.data_vars):
            _log.info(
                f"Variable {var} not present in gridded science ds. Skipping plots",
            )
            continue

        sci_timesection_plot(ds, var, base_path=base_path, show=show)
        sci_spatialsection_plot(ds, var, base_path=base_path, show=show)
        sci_spatialgrid_plot(ds, var, base_path=base_path, show=show)


def eng_tvt_loop(
    ds: xr.Dataset,
    base_path: str | None = None,
    show: bool = False,
):
    """
    A loop/wrapper function to:
    a) create the dictionary used for engineering thisvsthat plots and
    b) create said plots using the tvt function

    Arguments let the user specify if these plots should be saved, and/or shown

    Parameters
    ----------
    ds : xarray Dataset
        Timeseries engineering dataset
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        Nothing
    """

    _log.info("LOOP: making engineering tvt plots")
    eng_dict = eng_plots_to_make(ds)
    for key in eng_dict.keys():
        eng_tvt_plot(ds, eng_dict, key, base_path=base_path, show=show)


def sci_timeseries_loop(
    ds: xr.Dataset,
    base_path: str | None = None,
    show: bool = False,
):
    """
    A loop/wrapper function to use a timeseries science dataset to make plots
    of all sci_vars variables.
    Specifically, for each sci_var present in the dataset,
    create a timeseries plot

    Arguments let the user specify if these plots should be saved, and/or shown

    Parameters
    ----------
    ds : xarray Dataset
        Timeseries science dataset
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        Nothing
    """

    # plt.scatter(sci_ds_g.time, sci_ds_g.profile)

    _log.info("LOOP: making science timeseries plots")
    for var in sci_vars:
        _log.debug(f"var {var}")
        if var not in list(ds.data_vars):
            _log.info(
                f"Variable {var} not present in timeseries science ds. Skipping plots",
            )
            continue

        sci_timeseries_plot(ds, var, base_path=base_path, show=show)


def eng_timeseries_loop(
    ds: xr.Dataset,
    base_path: str | None = None,
    show: bool = False,
):
    """
    A loop/wrapper function to use a timeseries engineering dataset to make plots
    of all eng_vars variables.
    Specifically, for each eng_vars present in the dataset,
    create a timeseries plot

    Arguments let the user specify if these plots should be saved, and/or shown

    Parameters
    ----------
    ds : xarray Dataset
        Timeseries science dataset
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        Nothing
    """

    # plt.scatter(sci_ds_g.time, sci_ds_g.profile)
    _log.info("LOOP: making engineering timeseries plots")
    for var in eng_vars:
        _log.debug(f"var {var}")
        if var not in list(ds.data_vars):
            _log.info(
                f"Variable {var} not present in timeseries eng ds. Skipping plots",
            )
            continue

        eng_timeseries_plot(ds, var, base_path=base_path, show=show)


def sci_ts_loop(
    ds: xr.Dataset,
    base_path: str | None = None,
    show: bool = False,
):
    """
    A loop/wrapper function to use a timeseries science dataset to make plots
    of all sci_vars variables.
    Specifically, for each sci_vars present in the dataset,
    create a ts plot

    Arguments let the user specify if these plots should be saved, and/or shown

    Parameters
    ----------
    ds : xarray Dataset
        Timeseries science dataset
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        Nothing
    """

    _log.info("LOOP: making ts plots")
    for var in sci_vars:
        _log.debug(f"var {var}")
        if var not in list(ds.data_vars):
            _log.info(
                f"Variable {var} not present in timeseries sci ds. Skipping plots",
            )
            continue

        ts_plot(ds, var, base_path=base_path, show=show)


def sci_surface_map_loop(
    ds: xr.Dataset,
    crs,
    base_path: str | None = None,
    show: bool = False,
    bar: xr.Dataset | None = None,
    figsize_x = 8.5, 
    figsize_y = 11, 
):
    """
    A loop/wrapper function to use a timeseries science dataset to make plots
    of all sci_vars variables.
    Specifically, for each sci_vars present in the dataset,
    create a surface map using the bar dataset

    Arguments let the user specify if these plots should be saved, and/or shown

    Parameters
    ----------
    ds : xarray Dataset
        Gridded science dataset
    crs:
        cartopy.crs projection to be used for the map
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed
    bar : xarray Dataset
        ETOPO dataset with which to make contour lines

    Returns
    -------
        Nothing
    """

    _log.info("LOOP: making surface maps")
    for var in (sci_vars + ["profile_index"]):
        _log.debug(f"var {var}")
        if var not in list(ds.data_vars):
            _log.info(
                f"Variable {var} not present in timeseries sci ds. Skipping plots",
            )
            continue

        sci_surface_map(
            ds, var=var, crs=crs, base_path=base_path, show=show, bar=bar, 
            figsize_x=figsize_x, figsize_y=figsize_y, 
        )


def save_plot(
    fig: matplotlib.figure.Figure,
    file_dir: str,
    file_name: str,
):
    """
    Wrapper function to:
        Ensure 'file_dir' (a string) is a directory, and make it if necessary
        Save the matplotlib 'figure' object to 'file_name' (str) in 'file_dir'
    """
    # if not base_path is None:
    #     file_dir = os.path.join(base_path, "engineering", "thisVsThat")
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file_name)

    _log.debug(f"Saving {file_path}")
    fig.savefig(file_path)


def sci_timesection_plot(
    ds: xr.Dataset,
    var: str,
    base_path: str | None = None,
    show: bool = False,
):
    """
    Create timesection plots: variable plotted on time and depth
    Saves the plot to path: base_path/science/timeSections

    Parameters
    ----------
    ds : xarray dataset
        Gridded glider science dataset.
        This is intended to be a gridded dataset produced by slocum.binary_to_nc
    var : str
        The name of the variable to plot
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        matplotlib.Figure.figure object
    """

    if var not in list(ds.data_vars):
        _log.info(f"Variable name {var} not present in ds. Skipping plot")
        return

    _log.info(f"Making timesection plot for variable {var}")
    deployment = ds.deployment_name
    # glider = utils.split_deployment(deployment)[0]
    project = ds.project

    fig, ax = plt.subplots(figsize=(11, 8.5))
    std = np.nanstd(ds[var])
    mean = np.nanmean(ds[var])

    p1 = ax.pcolormesh(ds.time, ds.depth, adj_var(ds, var), cmap=sci_colors[var])
    fig.colorbar(p1).set_label(label=adj_var_label(ds, var), size=label_size)
    ax.invert_yaxis()

    ax.set_title(
        f"Deployment {deployment} for project {project}\n std={std:0.2f} mean={mean:0.2f}",
        size=14,
    )  # use set_title so that title is centered over the plot
    ax.set_xlabel("Time", size=label_size)
    ax.set_ylabel("Depth [m]", size=label_size)
    # t = ax.text(0, -0.18, caption, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, wrap=True)

    # for label in ax.get_xticklabels(which='major'):
    #     label.set(rotation=15, horizontalalignment='center')
    fig.autofmt_xdate()
    # fig_cnt += 1
    if base_path is not None:
        save_plot(
            fig,
            os.path.join(base_path, "science", "timeSections"),
            f"{deployment}_{var}_timesection.png",
        )

    if show:
        plt.show()
    plt.close(fig)

    return fig


def sci_spatialsection_plot(
    ds: xr.Dataset,
    var: str,
    base_path: str | None = None,
    show: bool = False,
):
    """
    Create spatialsection plots: variable plotted on lat or lon, and depth
    Saves the plot to path: base_path/science/spatialSections

    Parameters
    ----------
    ds : xarray dataset
        Gridded glider science dataset
    var : str
        The name of the variable to plot
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        matplotlib.Figure.figure object
    """

    if var not in list(ds.data_vars):
        _log.info(f"Variable name {var} not present in ds. Skipping plot")
        return

    _log.info(f"Making spatialsection plot for variable {var}")
    deployment = ds.deployment_name
    project = ds.project

    fig, axs = plt.subplots(1, 2, figsize=(11, 8.5), sharey=True)
    std = np.nanstd(ds[var])
    mean = np.nanmean(ds[var])

    ### Lon
    axs[0].pcolormesh(
        ds.longitude,
        ds.depth,
        adj_var(ds, var),
        cmap=sci_colors[var],
    )
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Longitude [Deg]", size=label_size)
    axs[0].set_ylabel("Depth [m]", size=label_size)
    axs[0].text(
        0.05,
        0.95,
        "A.",
        size=16,
        ha="left",
        fontweight="bold",
        transform=axs[0].transAxes,
        color="white",
        antialiased=True,
    )

    ### Lat
    p2 = axs[1].pcolormesh(
        ds.latitude,
        ds.depth,
        adj_var(ds, var),
        cmap=sci_colors[var],
    )
    # p2 = axs[1].pcolormesh(sci_ds_g.latitude, sci_ds_g.depth, sci_ds_g[var], cmap=sci_colors[var])
    fig.colorbar(p2).set_label(label=adj_var_label(ds, var), size=label_size)
    # axs[1].invert_yaxis()

    axs[1].set_xlabel("Latitude [Deg]", size=label_size)
    axs[1].text(
        0.05,
        0.95,
        "B.",
        size=16,
        ha="left",
        fontweight="bold",
        transform=axs[1].transAxes,
        color="white",
        antialiased=True,
    )
    # axs[1].set_ylabel(f"Depth [m]", size=14)

    fig.suptitle(
        f"Deployment {deployment} for project {project}\n std={std:0.2f} mean={mean:0.2f}",
        size=title_size,
    )

    # t = fig.text(0, -0.18, caption, horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes, wrap=True)

    for label in axs[0].get_xticklabels(which="major"):
        label.set(rotation=15, horizontalalignment="center")
    for label in axs[1].get_xticklabels(which="major"):
        label.set(rotation=15, horizontalalignment="center")
    # fig_cnt += 1

    if base_path is not None:
        save_plot(
            fig,
            os.path.join(base_path, "science", "spatialSections"),
            f"{deployment}_{var}_spatialSections.png",
        )

    if show:
        plt.show()
    plt.close(fig)

    return fig


def sci_spatialgrid_plot(
    ds: xr.Dataset,
    var: str,
    base_path: str | None = None,
    show: bool = False,
):
    """
    Create spatial grid plots: plot variable value by lat/lon/depth
    If specified, saves the plot to path: base_path/science/spatialGrids

    Parameters
    ----------
    ds : xarray dataset
        Gridded glider science dataset.
        This is intended to be a gridded dataset produced by slocum.binary_to_nc
    var : str
        The name of the variable to plot
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments.
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        matplotlib.Figure.figure object
    """

    _log.info(f"Making spatialgrid plot for variable {var}")
    gs = GridSpec(
        5,
        5,
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )
    deployment = ds.deployment_name
    project = ds.project

    fig = plt.figure(figsize=(11, 8.5))

    ax0 = fig.add_subplot(gs[0:3, 0:3])
    ax1 = fig.add_subplot(gs[3:5, 0:3])
    ax2 = fig.add_subplot(gs[0:3, 3:5])
    # std = np.nanstd(sci_ds_g[var])
    # mean = np.nanmean(sci_ds_g[var])

    # _,_,var_ = np.meshgrid(sci_ds_g.longitude.values, sci_ds_g.latitude.values, sci_ds_g[var].sel(depth=0, method='nearest'))
    # ax0.pcolormesh(sci_ds_g.longitude, sci_ds_g.latitude, var_[:,:,0], cmap=sci_colors[var])

    p = ax0.scatter(
        ds.longitude,
        ds.latitude,
        c=ds[var].sel(depth=0, method="nearest"),
        cmap=sci_colors[var],
    )

    ax0.set_ylabel("Latitude [Deg]", size=label_size)
    ax0.set_xticks([])
    ax0.set_xticklabels([])

    # ax0.scatter(sci_ds.longitude, sci_ds.latitude, c=sci_ds[var], cmap=sci_colors[var])
    ax1.pcolormesh(ds.longitude, ds.depth, adj_var(ds, var), cmap=sci_colors[var])
    ax1.set_ylabel("Depth [m]", size=label_size)
    ax1.set_xlabel("Longitude [Deg]", size=label_size)
    ax1.invert_yaxis()

    ax2.pcolormesh(
        ds.depth,
        ds.latitude,
        np.transpose(adj_var(ds, var).values),
        cmap=sci_colors[var],
    )
    ax2.set_xlabel("Depth [m]", size=label_size)
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    fig.colorbar(p, location="top", ax=[ax2, ax0]).set_label(
        label=adj_var_label(ds, var),
        size=label_size,
    )
    fig.suptitle(f"Deployment {deployment} for project {project}", size=title_size)

    if base_path is not None:
        save_plot(
            fig,
            os.path.join(base_path, "science", "spatialGrids"),
            f"{deployment}_{var}_spatialGrids.png",
        )

    if show:
        plt.show()
    plt.close(fig)

    return fig


def eng_plots_to_make(ds: xr.Dataset):
    """
    Create dictionary used to make engineering plots.
    This output is intended to be passed to eng_tvt_plot()

    ------
    Parameters

    ds : xarray dataset
        Timeseries glider engineering dataset.
        This is intended to be produced by slocum.binary_to_nc
    """
    plots_to_make = {
        "oilVol": {
            "X": ds["commanded_oil_volume"],
            "Y": [ds["measured_oil_volume"]],
            "C": ["C0"],
            "cb": False,
        },
        "diveEnergy": {
            "X": ds["total_num_inflections"],
            "Y": [ds["amphr"], ds["total_amphr"]],
            "C": ["C0", "C1"],
            "cb": False,
        },
        "diveDepth": {
            "X": ds["target_depth"],
            "Y": [ds["depth"]],
            "C": ["C0"],
            "cb": False,
        },
        "inflections": {
            "X": ds["total_num_inflections"],
            "Y": [ds["total_amphr"]],
            "C": ["C0"],
            "cb": False,
        },
        "diveAmpHr": {
            "X": ds["depth"],
            "Y": [ds["amphr"]],
            "C": ["C0"],
            "cb": False,
        },
        "leakDetect": {
            "X": ds["time"],
            "Y": [
                ds["leak_detect"].rolling(time=900).mean(),
                ds["leak_detect_forward"].rolling(time=900).mean(),
                ds["leak_detect_science"].rolling(time=900).mean(),
            ],
            "C": ["C0", "C1", "C2"],
            "cb": False,
        },
        "vacuumDepth": {
            "X": ds["time"],
            "Y": [ds["vacuum"]],
            "C": [ds["depth"]],
            "cb": True,
        },
    }

    return plots_to_make


def eng_tvt_plot(
    ds: xr.Dataset,
    eng_dict: dict,
    key: str,
    base_path: str | None = None,
    show: bool = False,
):
    """
    Creates 'this vs that' plots of engineering variables
    Saves the plot to path: base_path/engineering/thisVsThat

    Parameters
    ----------
    ds : xarray dataset
        Timeseries glider engineering dataset.
        This is intended to be produced by slocum.binary_to_nc
    eng_dict : dictionary
        Dictionary produced by eng_plots_to_make()
        Used by this function to get
    key : str
        The name of the variable (i.e., key from eng_dict) to plot
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        matplotlib.Figure.figure object
    """

    if key not in list(eng_dict.keys()):
        raise ValueError(f"Variable name {key} not present in eng_dict. Skipping plot")

    deployment = ds.deployment_name
    _log.info(f"Making tvt plot for dictionary key {key}")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    for i in range(len(eng_dict[key]["Y"])):
        if key == "oilVol":
            plot = ax.scatter(eng_dict[key]["X"], eng_dict[key]["Y"][i])
        else:
            plot = ax.scatter(
                eng_dict[key]["X"],
                eng_dict[key]["Y"][i],
                label=eng_dict[key]["Y"][i].name,
                c=eng_dict[key]["C"][i],
            )

        if eng_dict[key]["cb"]:
            fig.colorbar(plot)

    ax.set_xlabel(eng_dict[key]["X"].name, size=label_size)
    ax.set_ylabel(eng_dict[key]["Y"][0].name, size=label_size)

    if len(eng_dict[key]["C"]) > 1:
        ax.legend()

    if eng_dict[key]["X"].name == "time":
        fig.autofmt_xdate()

    if base_path is not None:
        save_plot(
            fig,
            os.path.join(base_path, "engineering", "thisVsThat"),
            f"{deployment}_{key}_engmisc.png",
        )

    if show:
        plt.show()
    plt.close(fig)

    return fig


def eng_timeseries_plot(
    ds: xr.Dataset,
    var: str,
    base_path: str | None = None,
    show: bool = False,
):
    """
    Create timeseries plots of engineering variables
    Saves the plot to path: base_path/engineering/timeSeries

    Parameters
    ----------
    ds : xarray dataset
        Timeseries glider engineering dataset.
        This is intended to be produced by slocum.binary_to_nc
    var : str
        The name of the variable to plot
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        matplotlib.Figure.figure object
    """

    if var not in list(ds.data_vars):
        _log.info(f"Variable name {var} not present in ds. Skipping plot")
        return

    _log.info(f"Making eng timeseries plot for variable {var}")
    deployment = ds.deployment_name
    project = ds.project

    fig, ax = plt.subplots(figsize=(11, 8.5))

    ax.set_xlabel("Time", size=label_size)
    ax.set_ylabel(adj_var_label(ds, var), size=label_size)
    # ax.invert_yaxis()
    ax.set_title(f"Deployment {deployment} for project {project}", size=title_size)

    ax.scatter(ds.time, ds[var], s=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    # fig.colorbar(p, location="right").set_label(adj_var_label(ds, var), size=label_size)
    fig.autofmt_xdate()

    if base_path is not None:
        save_plot(
            fig,
            os.path.join(base_path, "engineering", "timeSeries"),
            f"{deployment}_{var}_timeseries.png",
        )

    if show:
        plt.show()
    plt.close(fig)

    return fig


def sci_timeseries_plot(
    ds: xr.Dataset,
    var: str,
    base_path: str | None = None,
    show: bool = False,
):
    """
    Create timeseries plots of science variables
    Saves the plot to path: base_path/science/timeSeries

    Parameters
    ----------
    ds : xarray dataset
        Timeseries glider science dataset.
        This is intended to be a gridded dataset produced by slocum.binary_to_nc
    var : str
        The name of the variable to plot
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        matplotlib.Figure.figure object
    """

    if var not in list(ds.data_vars):
        _log.info(f"Variable name {var} not present in ds. Skipping plot")
        return

    _log.info(f"Making sci timeseries plot for variable {var}")
    deployment = ds.deployment_name
    project = ds.project

    fig, ax = plt.subplots(figsize=(11, 8.5))

    ax.set_xlabel("Time", size=label_size)
    ax.set_ylabel("Depth [m]", size=label_size)
    ax.invert_yaxis()
    ax.set_title(f"Deployment {deployment} for project {project}", size=title_size)

    p = ax.scatter(ds.time, ds.depth, c=adj_var(ds, var), cmap=sci_colors[var], s=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.colorbar(p, location="right").set_label(adj_var_label(ds, var), size=label_size)
    fig.autofmt_xdate()

    if base_path is not None:
        save_plot(
            fig,
            os.path.join(base_path, "science", "timeSeries"),
            f"{deployment}_{var}_timeseries.png",
        )

    if show:
        plt.show()
    plt.close(fig)

    return fig


def ts_plot(
    ds: xr.Dataset,
    var: str,
    base_path: str | None = None,
    show: bool = False,
):
    """
    Create ts plots of science variables
    Saves the plot to path: base_path/science/TS

    Parameters
    ----------
    ds : xarray dataset
        Timeseries glider science dataset.
        This is intended to be produced by slocum.binary_to_nc
    var : str
        The name of the variable to plot
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed

    Returns
    -------
        matplotlib.Figure.figure object
    """

    if var not in list(ds.data_vars):
        _log.info(f"Variable name {var} not present in ds. Skipping plot")
        return

    _log.info(f"Making ts plot for variable {var}")
    deployment = ds.deployment_name
    start = ds.deployment_start[0:10]
    end = ds.deployment_end[0:10]

    Sg, Tg, sigma = utils.ts_calculations(ds)

    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    C0 = ax.contour(Sg, Tg, sigma, colors="grey", zorder=1)
    plt.clabel(C0, colors="k", fontsize=9)
    p0 = ax.scatter(
        ds.salinity,
        ds.potential_temperature,
        c=adj_var(ds, var),
        cmap=sci_colors[var],
        s=5,
    )
    fig.colorbar(
        p0,
        orientation="vertical",
        location="right",
        shrink=1,
    ).set_label(label=adj_var_label(ds, var), size=label_size)

    ax.set_title(f"{deployment} from {start} to {end}", size=title_size)
    ax.set_xlabel("Salinity [PSU]", size=label_size)
    ax.set_ylabel("Potential temperature [°C]", size=label_size)

    # plt.savefig(f"{sci_save_path}/TS/{deployment}_{var}_tsPlot.png")
    if base_path is not None:
        save_plot(
            fig,
            os.path.join(base_path, "science", "TS"),
            f"{deployment}_{var}_tsPlot.png",
        )

    if show:
        plt.show()
    plt.close(fig)

    return fig


def crs_map(crs_str: str):
    """
    Map string crs_str to a Cartopy projection. 
    Returns a cartopy.crs class 

    This is convenience function, so that a user looking to use 
    sci_surface_map does not have to import cartopy in their script

    https://scitools.org.uk/cartopy/docs/latest/reference/projections.html
    """
    if crs_str == "PlateCarree":
        return ccrs.PlateCarree()
    elif crs_str == "Mercator":
        return ccrs.Mercator()
    else:
        raise ValueError(f"invalid crs_str: {crs_str}")


def sci_surface_map(
    ds: xr.Dataset,
    var: str,
    crs,
    base_path: str | None = None,
    show: bool = False,
    bar: xr.Dataset | None = None,
    figsize_x = 8.5, 
    figsize_y = 11, 
):
    """
    Create surface maps of science variables
    Saves the plot to path: base_path/science/maps

    Parameters
    ----------
    ds : xarray dataset
        Gridded science dataset
    var : str
        The name of the variable to plot
    crs : 
        An instantiated cartopy projection, such as cartopy.crs.PlateCarree()
        or cartopy.crs.Mercator()
        If crs is a string, then it is passed to plots.crs_map()
    base_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments
    show : bool
        Boolean indicating if the plots should be shown before being closed
    bar : xarray dataset
        dataset of ETOPO 15 arc-second global relief model.
        Likely downloaded from ERDDAP - must span glider lat/lon
        A future task is to make a function for pulling this file from ERDDAP, if necessary
        Eg: https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.nc?z%5B(30):1:(45)%5D%5B(-135):1:(-120)%5D

    Returns
    -------
        matplotlib.Figure.figure object
    """

    if var not in list(ds.data_vars):
        _log.info(f"Variable name {var} not present in ds. Skipping plot")
        return

    _log.info(f"Making surface map for variable {var}")
    deployment = ds.deployment_name
    start = ds.deployment_start[0:10]
    end = ds.deployment_end[0:10]

    map_lon_border = 0.1
    map_lat_border = 0.2
    glider_lon_min = ds.longitude.min()
    glider_lon_max = ds.longitude.max()
    glider_lat_min = ds.latitude.min()
    glider_lat_max = ds.latitude.max()

    # Using Cartopy
    if isinstance(crs, str):
        crs = crs_map(crs)
    fig, ax = plt.subplots(
        figsize=(figsize_x, figsize_y),
        subplot_kw={"projection": crs},
    )
    ax.set_xlabel("\n\n\nLongitude [Deg]", size=14)
    ax.set_ylabel("Latitude [Deg]\n\n\n", size=14)

    # Set extent of the map based on the glider data
    ax.set_extent(
        [
            glider_lon_min - map_lon_border,
            glider_lon_max + 3 * map_lon_border,
            glider_lat_min - map_lat_border,
            glider_lat_max + map_lat_border,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Add coastlines, countries, and continents
    # ax.coastlines(resolution="110m", linestyle="-", color="black")
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="#e0b479")
    ax.add_feature(cfeature.OCEAN, edgecolor="none", facecolor="#7bcbe3")

    # Add parallels and meridians; no scale bar since no built-in function
    gl = ax.gridlines(draw_labels=["bottom", "left"])
    gl.xlabel_style = {"rotation": 15}

    # Scatter plot data given data, colored by values in the top 10m
    p = ax.scatter(
        ds.longitude,
        ds.latitude,
        c=ds[var].where(ds.depth <= 10, drop=True).mean(dim="depth"),
        cmap=sci_colors[var],
        s=10,
        zorder=2.5,
        transform=ccrs.PlateCarree(),
    )

    if bar is not None:
        lon, lat = np.meshgrid(bar.z.lon, bar.z.lat)
        C0 = ax.contour(
            lon,
            lat,
            bar.z,
            levels=4,
            colors="grey",
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(C0, colors="grey", fontsize=9, inline=True)
        # m.contourf(lon, lat, bar.z, cmap="Pastel1")

    # Add colorbar
    fig.colorbar(p, ax=ax, shrink=0.6, location="right").set_label(
        label=adj_var_label(ds, var),
        size=label_size,
    )
    ax.set_title(
        f"{deployment}: 0 - 10m average {var}\nfrom {start} to {end}",
        size=title_size,
    )

    if base_path is not None:
        save_plot(
            fig,
            os.path.join(base_path, "science", "maps"),
            f"{deployment}_{var}_map_0-10.png",
        )

    if show:
        plt.show()
    plt.close(fig)

    return fig
