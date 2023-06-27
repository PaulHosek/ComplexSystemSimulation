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

        # calculate the expected values
        y_expect = D(np.log10(areas),*fit_params[:4])
    except: # RuntimeError:
        return  np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if len(areas > 7):
        Dims = []
        for i in range(3, len(areas)-3):
            dim = 2* (np.log(perimeters[i+3])-np.log(perimeters[i-3]))/(np.log(areas[i+3])-np.log(areas[i-3]))
            Dims.append(dim)
        
        return areas, y_expect, pcov, areas[3:-3], np.array(Dims)
    
    else:
        return  areas, y_expect, pcov, np.array([]), np.array([])

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

        plt.tight_layout()
        plt.savefig(f"experiments/{experiment_name}/figures/{run[0].replace('.npy','')}.png", dpi = 300)
