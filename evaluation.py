import numpy as np
from scipy.ndimage import label

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
    pond_clusters, _ = label(np.where(ponds == pond_val, 1, 0))

    clusters = np.unique(pond_clusters)[1:]
    length = len(clusters)
    areas = np.zeros(length)
    perimeters = np.zeros(length)
    dimensions = np.zeros(length)

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