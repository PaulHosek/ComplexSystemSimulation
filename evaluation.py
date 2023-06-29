import numpy as np
import scipy.ndimage as ndimage
from numba import njit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from matplotlib import colors
import re

# make a color map of fixed colors
cmap = colors.ListedColormap(['cyan', 'white', 'blue'])
bounds=[-100,0.5,3,100]
norm = colors.BoundaryNorm(bounds, cmap.N)


def perim_area(ponds, pond_val=-1, ice_val=1):
    '''
    Calculates the area and the perimeter of the meltpods

    Arguments:
        ponds -- 2D ndarray 

    Keyword Arguments:
        pond_val -- value indicating that cell is part of a melt pond (default: {-1})
        ice_val --  value indicating that cell is ice (default: {1})

    Returns:
        areas -- 1D ndarray with the area of each melt pond
        perimeters -- 1D ndarray with the perimeter of each melt pond
    '''

    # Get the pond clusters
    pond_clusters, _ = ndimage.label(np.where(ponds == pond_val, 1, 0))

    clusters = np.unique(pond_clusters)[1:]
    length = len(clusters)
    areas = np.zeros(length)
    perimeters = np.zeros(length)

    rows, cols = ponds.shape

    for i, cluster_id in enumerate(clusters):
        areas[i] = np.sum(pond_clusters == cluster_id)

        # Get the total number of neighboring cells for each cluster
        for cell in np.transpose(np.nonzero(pond_clusters == cluster_id)):
            row, col = cell

            # Calculate indices of neighboring cells with periodic boundaries
            neighbors = np.array([
                ponds[(row + 1) % rows, col],  # Right neighbor
                ponds[(row - 1) % rows, col],  # Left neighbor
                ponds[row, (col + 1) % cols],  # Down neighbor
                ponds[row, (col - 1) % cols],  # Up neighbor
            ])

            perimeters[i] += np.sum(neighbors == ice_val)

    return areas, perimeters


def detect_percolation(grid, sidelength):
    """
    Get the percolating cluster if it exists.

    :param grid: Input grid.
    :type grid: numpy.ndarray
    :param sidelength: Side length of the grid.
    :type sidelength: int
    :return: A tuple containing a boolean indicating whether a percolating cluster exists,
        and the masked array of the cluster if found, otherwise None.
    :rtype: tuple[bool, numpy.ndarray or None]
    """

    # 1. Identify and sort clusters by area; remove small clusters
    labels, num = ndimage.label(grid)
    area = ndimage.sum(grid, labels, index=np.arange(labels.max() + 1))
    unique, counts = np.unique(labels, return_counts=True)
    cluster_candidates = np.asarray((unique, counts)).T[1:] # index, area array
    cluster_candidates = cluster_candidates[cluster_candidates[:, 1] >= sidelength]
    cluster_candidates = cluster_candidates[cluster_candidates[:,1].argsort()][::-1]

    # Create dict with {cluster_idx: set of edges}
    top = np.unique(labels[0])
    right = np.unique(labels[:, -1])
    bottom = np.unique(labels[-1])
    left = np.unique(labels[:,0])
    cluster_edge = {}
    for border_idx, clusters in enumerate([top, right, bottom, left], start=1):
        clusters = set(clusters[clusters != 0])
        for c in clusters:
            cluster_edge.setdefault(c, set()).add(border_idx)

    # Iterate over clusters from largest to smallest and test if perturbation occurs.
    for cluster_id,_ in cluster_candidates:
        edges = cluster_edge.get(cluster_id,set())
        if edges.issuperset({1,3}) or edges.issuperset({2,4}):
            return True, np.ma.masked_where(labels != cluster_id,labels)

    return False, None

def fractal_dim(ponds, pond_val=-1, ice_val=1, bins = 50, min_area = 0):

    # get areas and perimeters
    areas, perimeters = perim_area(ponds, pond_val = pond_val, ice_val = ice_val)

    # sort arrays
    areas, perimeters = zip(*sorted(zip(areas, perimeters)))
    areas = np.array(areas)
    perimeters = np.array(perimeters)[areas >= min_area]
    areas = areas[areas >= min_area]

    # bin data and get the lowest perimeter for fitting
    areas, perimeters = get_lowest(areas, perimeters, bins = bins)

    try:
        # Perform curve fitting
        fit_params, pcov = curve_fit(integral_D, np.log10(areas), np.log10(perimeters), p0=None)
        areas_plot = 10**np.linspace(0,np.log10(areas.max()),1000)
        # calculate the expected values
        y_expect = D(np.log10(areas_plot),*fit_params[:4])
    except: # RuntimeError:
        return  np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if len(areas > 7):
        Dims = []
        for i in range(3, len(areas)-3):
            dim = 2* (np.log(perimeters[i+3])-np.log(perimeters[i-3]))/(np.log(areas[i+3])-np.log(areas[i-3]))
            Dims.append(dim)
        
        return areas_plot, y_expect, pcov, areas[3:-3], np.array(Dims)
    
    else:
        return areas, y_expect, pcov, np.array([]), np.array([])

# Define the function D(x) and its integral
def integral_D(x, a1, a2, a3, a4, a5):
    return (a1 / (2 * a2)) * np.log(np.cosh(a2 * (x - a3))) + (a4 / 2) * x + a5

def D(x, a1, a2, a3, a4):
    return a1 * np.tanh(a2 * (x - a3)) + a4

def get_lowest(areas_sorted, perimeters_sorted, bins=100):

    _ , area_bins = np.histogram(np.log10(areas_sorted), bins = bins)
    areas_binned = []
    min_perimeters = []

    for low, high in zip(area_bins[:-1], area_bins[1:]):
        
        # Filter the sorted perimeters based on the current bin's area range
        filtered_perimeters = perimeters_sorted[(np.log10(areas_sorted) >= low) & (np.log10(areas_sorted) < high)]

        # Check if the filtered perimeters array is non-empty
        if filtered_perimeters.size > 0:
            # Calculate the minimum perimeter for the current bin
            min_perimeter = np.min(filtered_perimeters)
            # Calculate the mean area for the current bin
            bin_area = np.mean([10**low, 10**high])
            areas_binned.append(bin_area)
            min_perimeters.append(min_perimeter)

    return np.array(areas_binned), np.array(min_perimeters)

# Define a function to extract the numeric part of the filename
def extract_number(filename):
    '''
    function to extract the numeric part of the filename

    Arguments:
        filename -- string

    Returns:
        integer in filename
    '''
    match = re.search(r"_i=(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1     

def make_plots(experiment_name, threshold = 0.01):

    #create figure folders
    if not os.path.exists(f"experiments/{experiment_name}/figures/"):
        os.mkdir(f"experiments/{experiment_name}/figures")

    h_filenames = sorted(os.listdir(f"experiments/{experiment_name}/pond"), key = extract_number)
    H_filenames = sorted(os.listdir(f"experiments/{experiment_name}/ice"), key = extract_number)

    ice_fraction = []
    pond_fraction = []
    ocean_fraction = []

    for run in zip(h_filenames, H_filenames):
        h = np.load(f"experiments/{experiment_name}/pond/{run[0]}")
        H = np.load(f"experiments/{experiment_name}/ice/{run[1]}")

        plot_array = np.where(H>0, 1, 5)
        plot_array = np.where(h>threshold, 0, plot_array)

        ice_fraction.append(np.sum(plot_array==1)/len(plot_array)**2)
        pond_fraction.append(np.sum(plot_array==0)/len(plot_array)**2)
        ocean_fraction.append(np.sum(plot_array==5)/len(plot_array)**2)

        if pond_fraction[-1] > 0:
            areas_dim, dimensions, _, areas_scatter, dimensions_scatter = fractal_dim(np.where(plot_array == 0, -1, 1), -1, 1, 50)

        plt.clf()
        # define figure layout
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 4, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[1, 2:4])

        ax1.imshow(plot_array, cmap=cmap, norm=norm)

        ax2.plot(ice_fraction, label = 'ice')
        ax2.plot(pond_fraction, label = 'pond')
        ax2.plot(ocean_fraction, label = 'ocean')
        ax2.set_ylim([0, 1])
        ax2.legend()
        
        if pond_fraction[-1] > 0:
            if areas_dim != np.array([]):
                ax3.plot(areas_dim, dimensions)
            if areas_scatter != np.array([]):
                ax3.scatter(areas_scatter, dimensions_scatter)
            
            ax3.set_xscale('log')
            ax3.set_xlabel('area [m^2]')
            ax3.set_ylabel('fractal dimension')
            ax3.set_ylim([.5, 3])

        plt.tight_layout()
        plt.savefig(f"experiments/{experiment_name}/figures/{run[0].replace('.npy','')}.png", dpi = 300)

def make_plots_no_fracking(experiment_name, threshold = 0.01):

    #create figure folders
    if not os.path.exists(f"experiments/{experiment_name}/figures/"):
        os.mkdir(f"experiments/{experiment_name}/figures")

    h_filenames = sorted(os.listdir(f"experiments/{experiment_name}/pond"), key = extract_number)
    H_filenames = sorted(os.listdir(f"experiments/{experiment_name}/ice"), key = extract_number)

    ice_fraction = []
    pond_fraction = []
    ocean_fraction = []

    for run in zip(h_filenames, H_filenames):
        h = np.load(f"experiments/{experiment_name}/pond/{run[0]}")
        H = np.load(f"experiments/{experiment_name}/ice/{run[1]}")

        plot_array = np.where(H>0, 1, 5)
        plot_array = np.where(h>threshold, 0, plot_array)

        ice_fraction.append(np.sum(plot_array==1)/len(plot_array)**2)
        pond_fraction.append(np.sum(plot_array==0)/len(plot_array)**2)
        ocean_fraction.append(np.sum(plot_array==5)/len(plot_array)**2)

        plt.clf()
        # define figure layout
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(3, 4, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0:2])
        ax2 = fig.add_subplot(gs[1, 2:4])

        ax1.imshow(plot_array, cmap=cmap, norm=norm)

        ax2.plot(ice_fraction, label = 'ice')
        ax2.plot(pond_fraction, label = 'pond')
        ax2.plot(ocean_fraction, label = 'ocean')
        ax2.set_title('area fractions')
        ax2.set_ylim([0, 1])
        ax2.legend(loc='center left')

        plt.tight_layout()
        plt.savefig(f"experiments/{experiment_name}/figures/{run[0].replace('.npy','')}.png", dpi = 300)

def bootstrapping(ponds, pond_val=-1, ice_val=1, num_bootstrap=100):
    """
    This function returns a list of areas and perimeters after bootstrapping has been applied.
    :param ponds:
    :param pond_val:
    :param ice_val:
    :param num_bootstrap:
    :return:
    """

    # get areas and perimeters
    areas, perimeters = perim_area(ponds, pond_val=pond_val, ice_val=ice_val)

    # Bootstrap resampling
    areas_bootstrap = []
    perimeters_bootstrap = []

    for _ in range(num_bootstrap):
        indices = np.random.choice(len(areas), size=len(areas), replace=True)
        areas_sampled = areas[indices]
        perimeters_sampled = perimeters[indices]

        areas_bootstrap.append(areas_sampled)
        perimeters_bootstrap.append(perimeters_sampled)

    return areas_bootstrap, perimeters_bootstrap


def inv_D(y, a1, a2, a3, a4):
    """
    This function returns the inverse of D.
    :param y:
    :param a1:
    :param a2:
    :param a3:
    :param a4:
    :return:
    """
    arg = (y-a4)/a1

    x = 10**np.arctanh(arg/a2) + 10**a3

    return x

def fractal_dim_from_ap(areas, perimeters, bins = 50, min_area=0):
    """
    This funtion returns, if exists, the inflection point for a given list of areas and perimeters.
    :param areas:
    :param perimeters:
    :param bins:
    :param min_area:
    :return:
    """

    # sort arrays
    areas, perimeters = zip(*sorted(zip(areas, perimeters)))
    areas = np.array(areas)
    perimeters = np.array(perimeters)[areas >= min_area]
    areas = areas[areas >= min_area]

    # bin data and get the lowest perimeter for fitting
    areas, perimeters = get_lowest(areas, perimeters, bins=bins)

    try:
        # Perform curve fitting
        fit_params, pcov = curve_fit(integral_D, np.log10(areas), np.log10(perimeters), p0=None)

        plot_areas = 10**np.linspace(0,5,1000)
        # calculate the expected values
        y_expect = D(np.log10(plot_areas), *fit_params[:4])

        point = (np.max(y_expect) + np.min(y_expect))/2

    except:  # RuntimeError:
        return None

    if len(areas > 7):
        Dims = []
        for i in range(3, len(areas) - 3):
            dim = 2 * (np.log(perimeters[i + 3]) - np.log(perimeters[i - 3])) / (
                        np.log(areas[i + 3]) - np.log(areas[i - 3]))
            Dims.append(dim)

        return point
    else:
        return None


def inflection_list(ponds):
    """
    This function calculates a list of mean inflection value.
    :param ponds:
    :return: mean inflection value.
    """

    areas_bootstrap, perimeters_bootstrap = bootstrapping(ponds,-1,1, 100)

    inflection_list = []

    for i in range(100):
        point = fractal_dim_from_ap(areas_bootstrap[i],perimeters_bootstrap[i], 50, 0)

        if point != None:
            inflection_list.append(point)

    return inflection_list