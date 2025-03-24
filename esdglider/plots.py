import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps 
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo

# import esdglider.utils as utils

_log = logging.getLogger(__name__)


def return_var(var):
    return var

def add_log(var, ds):
    return adjustments[var](ds[var])

def log_label(var):
    if var in adjustments_labels.keys():
        return f"{adjustments_labels[var]}({var} [{units[var]}])"
    else:
        return f"{var} [{units[var]}]"
    

adjustments = {
    "temperature": return_var, 
    "chlorophyll":np.log10, 
    "cdom":np.log10, 
    "backscatter_700":np.log10, 
    "oxygen_concentration":return_var,
    "salinity":return_var, 
    "density":return_var
}

adjustments_labels = {
    "chlorophyll":"$log_{10}$", 
    "cdom":"$log_{10}$", 
    "backscatter_700":"$log_{10}$", 
}

units = {"latitude":"deg",
    "longitude":"deg",
    "depth":"m",
    "heading":"deg",
    "pitch":"rad",
    "roll":"rad",
    "waypoint_latitude":"deg",
    "waypoint_longitude":"deg",
    "conductivity":"S$\\bullet m^{-1}$",
    "temperature":"°C",
    "pressure":"dbar",
    "chlorophyll":"µg•$L^{-l}$",
    "cdom":"ppb",
    "backscatter_700":"$m^{-1}$",
    "oxygen_concentration":"µmol•$L^{-1}$",
    "depth_ctd": "m",
    "distance_over_ground":"km",
    "salinity":"PSU",
    "potential_density":"kg $\\bullet m^{-3}$",
    "density":"kg $\\bullet m^{-3}$",
    "potential_temperature":"°C",
    "profile_index":"1",
    "profile_direction":"1"}

sci_colors = {
    "cdom":cmo.solar, 
    "chlorophyll": cmo.algae, 
    "oxygen_concentration":cmo.tempo, 
    "backscatter_700":colormaps['terrain'], 
    "temperature":cmo.thermal, 
    "potential_temperature":cmo.thermal, 
    "salinity":cmo.haline, 
    "density":colormaps['cividis'], 
    "potential_density":colormaps['cividis']
}


# def get_path_plots():
#     """
#     Return a dictionary of plotting-related paths

#     -----
#     Parameters



    
#     -----
#     Returns:
#         A dictionary with the relevant paths
#     """

#     path_base = "/home/sam_woodman_noaa_gov/deployment_plots"
#     sci_save_path = os.path.join(path_base, deployment, "science")
#     eng_save_path = os.path.join(path_base, deployment, "engineering")

#     # sci_save_path = f"/Users/cflaim/Documents/deployment_reports/{project}/{deployment}/plots/science/"
#     # eng_save_path = f"/Users/cflaim/Documents/deployment_reports/{project}/{deployment}/plots/engineering/"
#     sci_dirs_to_check = ["TS", "timeSections", "spatialSections", "maps", "miscPlots", "timeSeries"]
#     eng_dirs_to_check = ["timeSeries", "thisVsThat"]


#     return {

#     }


def plot_timesection(ds, var, plots_path=None):
    """
    Create timesection plots
    Saves the plot to path: plots_path/science/timeSections

    ------
    Parameters

    ds : xarray dataset
        Gridded glider science dataset. 
        Must contain 'deployment_name' and 'project' attributes.
        This is intended to be a gridded dataset produced by slocum.binary_to_nc

    var : str
        The name of the variable to plot

    plots_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments

    ------
    Returns
        matplotlib plt object
    """

    if not var in list(ds.data_vars):
        _log.info(f"Variable name {var} not present in ds. Skipping plot")
        return 

    deployment = ds.deployment_name
    # glider = utils.split_deployment(deployment)[0]
    project = ds.project

    fig, ax = plt.subplots(figsize=(11, 8.5))
    std = np.nanstd(ds[var])
    mean = np.nanmean(ds[var])    

    #     caption = f"Figure {fig_cnt}: Colorized {var} [{units[var]}] plotted with time on the x-axis and depth on the y-axis. \
    # {var[0].upper()}{var[1:]} was obsrved to have a minimum of {sci_ds_g[var].min():0.2f} {units[var]}, \
    # maximum of {sci_ds_g[var].max():0.2f} {units[var]}, mean of {mean:0.2f} {units[var]}, and standard \
    # deviation of {std:0.2f}. These data were collected by a Teledyne Slocum g3 glider named {glider} off of {location} \
    # from {pd.to_datetime(sci_ds_g.time.values.min()).strftime("%Y-%m-%d")} to {pd.to_datetime(sci_ds_g.time.values.max()).strftime("%Y-%m-%d")}. \
    # These data are spatially bound by {sci_ds_g.longitude.min():0.3f}°W, {sci_ds_g.longitude.max():0.3f}°W, {sci_ds_g.latitude.min():0.3f}°N, and {sci_ds_g.latitude.max():0.3f}°N."

    # if "700" in var:
    #     ax.pcolormesh(sci_ds_g.time, sci_ds_g.density, sci_ds_g[var]*1e10, cmap=sci_colors[var])

    p1 = ax.pcolormesh(ds.time, ds.depth, add_log(var, ds), cmap=sci_colors[var])
    fig.colorbar(p1).set_label(label=log_label(var), size=14)
    ax.invert_yaxis()

    ax.set_title(f"Deployemnt {deployment} for project {project}\n std={std:0.2f} mean={mean:0.2f}")
    ax.set_xlabel(f"Time", fontsize=14)
    ax.set_ylabel(f"Depth [m]", fontsize=14)
    # t = ax.text(0, -0.18, caption, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, wrap=True)

    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=15, horizontalalignment='center')
    # fig_cnt += 1
    if not plots_path is None:
        path_file = os.path.join(
            plots_path, "science", "timeSections", 
            f"{deployment}_{var}_timesection.png"
        )
        plt.savefig(path_file)

    # plt.show()

    return plt


def plot_spatialsection(ds, var, plots_path=None):
    """
    Create spatialsection plots
    Saves the plot to path: plots_path/science/spatialSections

    ------
    Parameters

    ds : xarray dataset
        Gridded glider science dataset. 
        Must contain 'deployment_name' and 'project' attributes.
        This is intended to be a gridded dataset produced by slocum.binary_to_nc

    var : str
        The name of the variable to plot

    plots_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments

    ------
    Returns
        matplotlib plt object
    """

    if not var in list(ds.data_vars):
        _log.info(f"Variable name {var} not present in ds. Skipping plot")
        return 

    deployment = ds.deployment_name
    project = ds.project

    fig, axs = plt.subplots(1,2,figsize=(11, 8.5), sharey=True)
    std = np.nanstd(ds[var])
    mean = np.nanmean(ds[var])

#     caption = f"Figure {fig_cnt}: A.) Colorized {var} [{units[var]}] plotted with longitude on the x-axis and depth on the y-axis. \
# B.) Colorized {var} [{units[var]}] plotted with latitude on the x-axis and depth on the y-axis. \
# {var[0].upper()}{var[1:]} was obsrved to have a minimum of {sci_ds_g[var].min():0.2f} {units[var]}, \
# maximum of {sci_ds_g[var].max():0.2f} {units[var]}, mean of {mean:0.2f} {units[var]}, and standard \
# deviation of {std:0.2f}. These data were collected by a Teledyne Slocum g3 glider named {glider} off of {location} \
# from {pd.to_datetime(sci_ds_g.time.values.min()).strftime("%Y-%m-%d")} to {pd.to_datetime(sci_ds_g.time.values.max()).strftime("%Y-%m-%d")}. \
# These data are spatially bound by {sci_ds_g.longitude.min():0.3f}°W, {sci_ds_g.longitude.max():0.3f}°W, {sci_ds_g.latitude.min():0.3f}°N, and {sci_ds_g.latitude.max():0.3f}°N."
        
    ### Lon
    p1 = axs[0].pcolormesh(ds.longitude, ds.depth, add_log(var, ds), cmap=sci_colors[var])
    # p1 = axs[0].pcolormesh(sci_ds_g.longitude, sci_ds_g.depth, sci_ds_g[var], cmap=sci_colors[var])

    # cbar = fig.colorbar(p1).set_label(label=f"{var} [{units[var]}]", size=14)
    axs[0].invert_yaxis()

    axs[0].set_xlabel(f"Longitude [Deg]", fontsize=14)
    axs[0].set_ylabel(f"Depth [m]", fontsize=14)
    axs[0].text(0.05, 0.95, "A.", fontsize=16, ha='left', fontweight="bold", 
                transform=axs[0].transAxes, color="white", antialiased=True)
    
    ### Lat
    p2 = axs[1].pcolormesh(ds.latitude, ds.depth, add_log(var, ds), cmap=sci_colors[var])
    # p2 = axs[1].pcolormesh(sci_ds_g.latitude, sci_ds_g.depth, sci_ds_g[var], cmap=sci_colors[var])
    fig.colorbar(p2).set_label(label=log_label(var), size=14)
    # axs[1].invert_yaxis()

    axs[1].set_xlabel(f"Latitude [Deg]", fontsize=14)
    axs[1].text(0.05, 0.95, "B.", fontsize=16, ha='left', fontweight="bold", 
                transform=axs[1].transAxes, color="white", antialiased=True)
    # axs[1].set_ylabel(f"Depth [m]", fontsize=14)

    fig.suptitle(f"Deployemnt {deployment} for project {project}\n std={std:0.2f} mean={mean:0.2f}")

    # t = fig.text(0, -0.18, caption, horizontalalignment='left', verticalalignment='center', transform=axs[0].transAxes, wrap=True)

    for label in axs[0].get_xticklabels(which='major'):
        label.set(rotation=15, horizontalalignment='center')
    # fig_cnt += 1

    if not plots_path is None:
        path_file = os.path.join(
            plots_path, "science", "spatialSections", 
            f"{deployment}_{var}_spatialSections.png"
        )
        plt.savefig(path_file)
        
    # plt.savefig(f"{sci_save_path}/spatialSections/{deployment}_{var}_spatialSections.png")
    # plt.show()
    return plt


def plot_spatialgrid(ds, var, plots_path=None):
    """
    Create spatial grid plots: plot variable value by lat/lon/depth

    ------
    Parameters

    ds : xarray dataset
        Gridded glider science dataset. 
        Must contain 'deployment_name' and 'project' attributes.
        This is intended to be a gridded dataset produced by slocum.binary_to_nc

    var : str
        The name of the variable to plot

    plots_path : str
        The 'base' of the plot path. If None, then the plot will not be saved
        Intended to be the 'plotdir' output of slocum.get_path_deployments.
        Currently ignored

    ------
    Returns
        matplotlib plt object (fig)
    """

    gs = GridSpec(5, 5,left=0.1, right=0.9, bottom=0.1, top=0.9,wspace=0.05, hspace=0.05)

    fig = plt.figure(figsize=(11, 8.5))

    ax0 = fig.add_subplot(gs[0:3, 0:3])
    ax1 = fig.add_subplot(gs[3:5, 0:3])
    ax2 = fig.add_subplot(gs[0:3, 3:5])
    # std = np.nanstd(sci_ds_g[var])
    # mean = np.nanmean(sci_ds_g[var])
    
    # _,_,var_ = np.meshgrid(sci_ds_g.longitude.values, sci_ds_g.latitude.values, sci_ds_g[var].sel(depth=0, method='nearest'))
    # ax0.pcolormesh(sci_ds_g.longitude, sci_ds_g.latitude, var_[:,:,0], cmap=sci_colors[var])
    
    p = ax0.scatter(
        ds.longitude, ds.latitude, c=ds[var].sel(depth=0, method='nearest'), 
        cmap=sci_colors[var])

    ax0.set_ylabel("Latitude [Deg]")
    ax0.set_xticks([])
    ax0.set_xticklabels([])

    # ax0.scatter(sci_ds.longitude, sci_ds.latitude, c=sci_ds[var], cmap=sci_colors[var])
    ax1.pcolormesh(ds.longitude, ds.depth, add_log(var, ds), cmap=sci_colors[var])
    ax1.set_ylabel("Depth [m]")
    ax1.set_xlabel("Longitude [Deg]")


    ax1.invert_yaxis()

    ax2.pcolormesh(
        ds.depth, ds.latitude, np.transpose(add_log(var, ds).values), 
        cmap=sci_colors[var])
    ax2.set_xlabel("Depth [m]")
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    fig.colorbar(p, location='top', ax=[ax2, ax0])

    return plt
